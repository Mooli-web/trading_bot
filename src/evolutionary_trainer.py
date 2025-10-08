# src/evolutionary_trainer.py (Final version - Using joblib's default robust backend)
import sys
import os
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
import shutil
from copy import deepcopy
import signal
from typing import Optional, List, Dict
import random
import multiprocessing as mp
import traceback
import threading
import time # <--- ماژول زمان اضافه شد

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.append(project_root)

from src.env_wrapper import RustEnvWrapper
from sb3_contrib import RecurrentPPO
from src.evaluation import evaluate_signals
from src.genetic_algorithm import tournament_selection, crossover, mutate, get_random_value, sanitize_and_align_hyperparams
from config.settings import Evolutionary, Training, WalkForward
from src.callbacks import TradingMetricsCallback
from src.indicator_calculator import IndicatorCalculator
from src.live_monitor import LiveMonitor

logger = logging.getLogger(__name__)

def _get_fitness_score(all_results: List[Dict]) -> float:
    all_pnl = [p for res in all_results for p in res.get('pnl_history', [])]
    if len(all_pnl) < 20: return -10000.0
    pnl_series = pd.Series(all_pnl)
    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else 999.0
    sharpe_ratio = pnl_series.mean() / pnl_series.std() if pnl_series.std() > 1e-9 else 0.0
    equity_curve = pnl_series.cumsum()
    max_drawdown = (equity_curve.cummax() - equity_curve).max()
    
    score = (profit_factor - 1.0) * 100
    if score > 0: score *= (1 + sharpe_ratio)
    score -= (max_drawdown / 100)**2 * 10
    
    return score if np.isfinite(score) else -10000.0

class Agent:
    def __init__(self, agent_id, hyperparams):
        self.id = agent_id; self.hyperparams = hyperparams
        self.model = None; self.fitness = -np.inf
        self.results_list = []; self.model_path = None
        self.key_metrics = {}

def run_agent_process(agent: Agent, train_df: pd.DataFrame, validation_df: pd.DataFrame, indicator_calculator: IndicatorCalculator, fold_model_path: str, eval_start_points: List[int], update_queue: mp.Queue) -> Agent:
    start_time = time.time()
    try:
        update_queue.put({'id': agent.id, 'status': 'Initializing'})
        
        train_env = RustEnvWrapper(df=train_df, indicator_calculator=indicator_calculator, agent_hyperparams=agent.hyperparams['rl_params'])
        callback = TradingMetricsCallback(agent_id=agent.id, update_queue=update_queue)

        rl_params = agent.hyperparams['rl_params'].copy()
        env_specific_params = ['stop_loss_atr_multiplier', 'take_profit_atr_multiplier', 'slippage_atr_fraction', 'commission_pips', 'swap_pips_per_day']
        for key in env_specific_params:
            rl_params.pop(key, None)
        
        model = RecurrentPPO("MultiInputLstmPolicy", train_env, verbose=0, **rl_params)
        model.learn(total_timesteps=Training.TOTAL_TIMESTEPS_PER_AGENT, callback=callback)

        update_queue.put({'id': agent.id, 'status': 'Evaluating'})
        all_results = [evaluate_signals(model=model, eval_data=validation_df.copy(), indicator_calculator=indicator_calculator, 
                                        agent_hyperparams=agent.hyperparams['rl_params'], agent_id=agent.id, start_step=start)
                       for start in eval_start_points]

        agent.results_list = all_results
        agent.fitness = _get_fitness_score(all_results)
        
        # --- [FIX] منطق محاسبه شاخص‌های نهایی اصلاح شد ---
        all_pnl = [p for res in all_results for p in res.get('pnl_history', [])]
        all_trades_info = [trade for res in all_results for trade in res.get('trade_history', [])]

        # ۱. مقداردهی اولیه تمام شاخص‌ها با مقادیر پیش‌فرض
        agent.key_metrics['trades'] = len(all_pnl)
        agent.key_metrics['win_rate'] = 0.0
        agent.key_metrics['profit_factor'] = 0.0
        agent.key_metrics['sharpe'] = 0.0
        agent.key_metrics['drawdown'] = 0.0
        agent.key_metrics['avg_duration'] = 0.0

        # ۲. تنها در صورت وجود معامله، مقادیر واقعی را محاسبه و جایگزین کن
        if all_pnl:
            pnl_series = pd.Series(all_pnl)
            agent.key_metrics['win_rate'] = (pnl_series > 0).mean() * 100
            
            gross_profit = pnl_series[pnl_series > 0].sum()
            gross_loss = abs(pnl_series[pnl_series < 0].sum())
            agent.key_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 1e-9 else 999.0
            
            if len(pnl_series) > 1:
                agent.key_metrics['sharpe'] = pnl_series.mean() / pnl_series.std() if pnl_series.std() > 1e-9 else 0.0
                equity_curve = pnl_series.cumsum()
                agent.key_metrics['drawdown'] = (equity_curve.cummax() - equity_curve).max()
            
            if all_trades_info:
                total_duration = sum(trade['duration_candles'] for trade in all_trades_info)
                agent.key_metrics['avg_duration'] = total_duration / len(all_trades_info)


        model_path = os.path.join(fold_model_path, f"{agent.id}.zip")
        model.save(model_path)
        agent.model_path = model_path
        
    except Exception as e:
        logger.error(f"A critical error occurred in agent {agent.id} process: {e}")
        logger.error(traceback.format_exc())
        agent.fitness = -float('inf')
        
    finally:
        duration = time.time() - start_time
        update_queue.put({
            'id': agent.id, 
            'status': 'Done', 
            'final_metrics': agent.key_metrics, 
            'fitness': agent.fitness,
            'duration': duration,
        })
    return agent

class EvolutionaryTrainer:
    def __init__(self, train_df, validation_df, indicator_calculator, base_hyperparams, monitor: LiveMonitor, resume=False, fold_number=1, initial_population=None):
        self.train_df = train_df; self.validation_df = validation_df
        self.indicator_calculator = indicator_calculator; self.base_hyperparams = base_hyperparams
        self.monitor = monitor; self.fold_number = fold_number
        self.population = []; self.generation = 0
        self.best_agent_overall = None; self.elites = []
        self.state_file_path = Training.STATE_FILE_PATH
        self.fold_model_path = os.path.join(Training.MODEL_SAVE_PATH, f"fold_{self.fold_number}")
        
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        
        if resume and os.path.exists(self.state_file_path):
            self._load_state()
        else:
            self._cleanup_fold_files()
            if initial_population:
                self.population = self._create_population_from_elites(initial_population)
            else:
                self.population = self._initialize_population()
            
    def run(self):
        self.population = sanitize_and_align_hyperparams(self.population, self.base_hyperparams)

        standardized_eval_points = [np.random.randint(low=49, high=len(self.validation_df) - 100) for _ in range(Evolutionary.NUM_EVAL_RUNS)]
        logger.info(f"Using standardized evaluation start points for Fold {self.fold_number}: {standardized_eval_points}")

        for gen in range(self.generation, Evolutionary.NUM_GENERATIONS):
            self.generation = gen
            self.monitor.start_generation(gen + 1, Evolutionary.NUM_GENERATIONS, self.population)
            
            with mp.Manager() as manager:
                update_queue = manager.Queue()
                
                monitor_thread = threading.Thread(
                    target=self.monitor.process_updates,
                    args=(update_queue, len(self.population)),
                    daemon=True
                )
                monitor_thread.start()

                job_results = Parallel(n_jobs=Evolutionary.N_JOBS)(
                    delayed(run_agent_process)(
                        agent, self.train_df, self.validation_df, self.indicator_calculator, 
                        self.fold_model_path, standardized_eval_points, update_queue
                    ) for agent in self.population
                )
                
                monitor_thread.join(timeout=5.0)

            successful_agents = [agent for agent in job_results if agent and agent.fitness > -9000.0]

            if not successful_agents:
                self.monitor.log_generation_summary(None, -1, "No successful agents. Re-initializing population.")
                self.population = self._initialize_population()
                self._save_state()
                continue
            
            successful_agents.sort(key=lambda x: x.fitness, reverse=True)
            best_gen_agent = successful_agents[0]
            avg_fitness = np.mean([a.fitness for a in successful_agents])
            self.monitor.log_generation_summary(best_gen_agent, avg_fitness)
            
            self.elites = successful_agents[:Evolutionary.ELITISM_COUNT]
            if not self.best_agent_overall or best_gen_agent.fitness > self.best_agent_overall.fitness:
                self.best_agent_overall = deepcopy(best_gen_agent)
            
            # --- Create the next generation ---
            next_gen = [deepcopy(e) for e in self.elites]
            for i, elite in enumerate(next_gen):
                elite.id = f"f{self.fold_number}_gen{gen+1}_agent{i}_elite"
            
            num_new_random = Evolutionary.get('NUM_RANDOM_AGENTS', 1)
            num_children = Evolutionary.POPULATION_SIZE - len(self.elites) - num_new_random

            for i in range(num_children):
                p1 = tournament_selection(successful_agents)
                p2 = tournament_selection(successful_agents)
                child_params = mutate(crossover(p1, p2), self.base_hyperparams, gen, Evolutionary.NUM_GENERATIONS)
                next_gen.append(Agent(f"f{self.fold_number}_gen{gen+1}_agent{len(self.elites)+i}", child_params))

            for i in range(num_new_random):
                random_params = {'rl_params': {p: get_random_value(v) for p, v in self.base_hyperparams['param_ranges'].items()}}
                next_gen.append(Agent(f"f{self.fold_number}_gen{gen+1}_agent{len(self.elites)+num_children+i}_random", random_params))

            self.population = next_gen
            self._save_state()
            
        if self.best_agent_overall and self.best_agent_overall.model_path and self.fold_number == WalkForward.NUM_SPLITS:
            final_model_path = os.path.join(Training.MODEL_SAVE_PATH, "final_model.zip")
            shutil.copy(self.best_agent_overall.model_path, final_model_path)
            logger.info(f"Final model from last fold saved to: {final_model_path}")

    def _initialize_population(self, num_agents=None) -> list:
        num_agents = num_agents or Evolutionary.POPULATION_SIZE
        generation_id = self.generation
        return [Agent(f"f{self.fold_number}_gen{generation_id}_agent{i}", {'rl_params': {p: get_random_value(v) for p, v in self.base_hyperparams['param_ranges'].items()}}) for i in range(num_agents)]
    
    def _create_population_from_elites(self, elites: list) -> list:
        next_gen = [deepcopy(e) for e in elites]
        for i, elite in enumerate(next_gen):
            elite.id = f"f{self.fold_number}_gen0_agent{i}_elite_carryover"
        
        num_to_create = Evolutionary.POPULATION_SIZE - len(elites)
        for i in range(num_to_create):
            parent = random.choice(elites)
            child_params = mutate(deepcopy(parent.hyperparams), self.base_hyperparams, 0, 1)
            next_gen.append(Agent(f"f{self.fold_number}_gen0_agent{len(elites)+i}_mutated", child_params))
        return next_gen

    def _save_state(self):
        for agent in self.population + ([self.best_agent_overall] if self.best_agent_overall else []):
            agent.model = None
        state = {
            'fold': self.fold_number, 
            'generation': self.generation, 
            'population': self.population, 
            'best_agent_overall': self.best_agent_overall
        }
        dump(state, self.state_file_path)

    def _load_state(self):
        try:
            state = load(self.state_file_path)
            if state.get('fold') != self.fold_number:
                self.population = self._initialize_population()
                return
            self.generation = state['generation']
            self.population = state['population']
            self.best_agent_overall = state['best_agent_overall']
            logger.info(f"Resuming Fold {self.fold_number} from Generation {self.generation}.")
        except Exception:
            logger.error("Failed to load state. Starting fresh.", exc_info=True)
            self.population = self._initialize_population()

    def _handle_exit(self, signum, frame):
        logger.info("Exit signal received. Saving state...")
        self._save_state()
        sys.exit(0)
        
    def get_elites(self) -> list:
        return self.elites
        
    def _cleanup_fold_files(self):
        if os.path.exists(self.fold_model_path):
            shutil.rmtree(self.fold_model_path)
        os.makedirs(self.fold_model_path, exist_ok=True)