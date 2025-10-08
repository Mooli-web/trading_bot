# src/evaluation.py (Final version with Terminal-based Live Monitoring)
import sys
import os
import logging
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Dict, Any, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.env_wrapper import RustEnvWrapper
from src.indicator_calculator import IndicatorCalculator

logger = logging.getLogger(__name__)

def evaluate_signals(
    model: BaseAlgorithm,
    eval_data: pd.DataFrame,
    indicator_calculator: IndicatorCalculator,
    agent_hyperparams: Dict[str, Any],
    agent_id: str = 'N/A',
    start_step: Optional[int] = None
) -> Dict[str, Any]:
    if eval_data.empty:
        logger.warning(f"Agent {agent_id}: Evaluation dataframe is empty. Skipping.")
        return {}

    try:
        env = RustEnvWrapper(
            df=eval_data, indicator_calculator=indicator_calculator,
            agent_hyperparams=agent_hyperparams
        )
    except Exception as e:
        logger.error(f"Agent {agent_id}: Failed to create Rust env for eval. Error: {e}", exc_info=True)
        return {}

    obs, _ = env.reset(start_step=start_step)
    done = False
    lstm_states = None
    
    while not done:
        try:
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except Exception as e:
            logger.error(f"Agent {agent_id}: Error during evaluation step. Halting. Error: {e}", exc_info=True)
            break

    return env.get_final_stats()