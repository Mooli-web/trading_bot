# src/genetic_algorithm.py (Final version with diversity enhancement)
import sys
import os
import random
import numpy as np
from copy import deepcopy
from typing import Dict, Any, List, TYPE_CHECKING
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.settings import Evolutionary

if TYPE_CHECKING:
    from src.evolutionary_trainer import Agent

logger = logging.getLogger(__name__)

def get_random_value(param_info: Dict[str, Any]):
    p_type = param_info['type']
    
    if p_type == 'choice':
        return random.choice(param_info['values'])
    
    p_range = param_info.get('range')
    if p_range is None:
        raise ValueError(f"Parameter with type '{p_type}' is missing a 'range'.")

    use_log_scale = param_info.get('log', False)

    if use_log_scale:
        if not (p_range[0] > 0 and p_range[1] > 0):
             raise ValueError("Log scale requires range values to be positive.")
        low = np.log10(p_range[0])
        high = np.log10(p_range[1])
        value = 10**random.uniform(low, high)
    else:
        value = random.uniform(p_range[0], p_range[1])
    
    if p_type == 'int':
        return int(round(value))
    
    return value

def sanitize_and_align_hyperparams(
    population: List['Agent'],
    base_hyperparams: Dict[str, Any]
) -> List['Agent']:
    sanitized_population = []
    
    expected_rl_keys = set(base_hyperparams.get('param_ranges', {}).keys())
    
    for agent in population:
        current_rl_params = agent.hyperparams.get('rl_params', {})
        
        for key in expected_rl_keys:
            if key not in current_rl_params:
                logger.warning(f"Agent {agent.id} was missing new RL param '{key}'. Adding a random value.")
                param_info = base_hyperparams['param_ranges'][key]
                current_rl_params[key] = get_random_value(param_info)
        
        current_rl_keys = set(current_rl_params.keys())
        obsolete_rl_keys = current_rl_keys - expected_rl_keys
        for key in list(obsolete_rl_keys):
            logger.warning(f"Obsolete RL param '{key}' removed from agent {agent.id}.")
            del current_rl_params[key]
            
        agent.hyperparams['rl_params'] = current_rl_params
        sanitized_population.append(agent)
        
    return sanitized_population

def tournament_selection(population: List['Agent']) -> 'Agent':
    if not population:
        raise ValueError("Cannot perform tournament selection on an empty population.")

    tournament_size = min(Evolutionary.TOURNAMENT_SIZE, len(population))
    tournament_contenders = random.sample(population, tournament_size)
    
    # [FIX] Introduce a chance to select a random contender to increase diversity
    if random.random() < Evolutionary.get('TOURNAMENT_RANDOM_CHANCE', 0.1):
        return deepcopy(random.choice(tournament_contenders))

    best_contender = max(tournament_contenders, key=lambda agent: agent.fitness)
    return deepcopy(best_contender)

def crossover(parent1_agent: 'Agent', parent2_agent: 'Agent') -> Dict[str, Any]:
    if random.random() > Evolutionary.CROSSOVER_PROBABILITY:
        return deepcopy(random.choice([parent1_agent.hyperparams, parent2_agent.hyperparams]))

    child_params = {'rl_params': {}}
    params1 = parent1_agent.hyperparams
    params2 = parent2_agent.hyperparams

    for key in params1['rl_params']:
        if key in params2['rl_params']:
            child_params['rl_params'][key] = deepcopy(random.choice([params1['rl_params'][key], params2['rl_params'][key]]))
        else:
            child_params['rl_params'][key] = deepcopy(params1['rl_params'][key])
    
    return child_params

def mutate(hyperparams: Dict[str, Any], base_hyperparams: Dict[str, Any], current_gen: int, total_gens: int) -> Dict[str, Any]:
    mutated_params = deepcopy(hyperparams)

    progress = current_gen / max(1, total_gens - 1)
    mutation_prob = Evolutionary.MUTATION_PROBABILITY_START * ((Evolutionary.MUTATION_PROBABILITY_END / Evolutionary.MUTATION_PROBABILITY_START) ** progress)
    strong_mutation_prob = Evolutionary.STRONG_MUTATION_PROBABILITY * (1 - progress)

    # Mutate RL parameters
    for key, value in mutated_params['rl_params'].items():
        if key in base_hyperparams['param_ranges'] and random.random() < mutation_prob:
            param_info = base_hyperparams['param_ranges'][key]
            
            if random.random() < strong_mutation_prob:
                mutated_params['rl_params'][key] = get_random_value(param_info)
            else:
                if param_info['type'] == 'choice':
                    mutated_params['rl_params'][key] = get_random_value(param_info)
                elif isinstance(value, (float, int)) and 'range' in param_info:
                    nudge_factor = 1.0 + (0.2 * (1 - progress))
                    new_value = value * random.uniform(1/nudge_factor, nudge_factor)
                    
                    clipped_value = np.clip(new_value, param_info['range'][0], param_info['range'][1])
                    
                    if param_info['type'] == 'int':
                        mutated_params['rl_params'][key] = int(round(clipped_value))
                    else:
                        mutated_params['rl_params'][key] = clipped_value

    # Final sanity checks for critical parameters
    rl = mutated_params['rl_params']
    if 'gamma' in rl: rl['gamma'] = np.clip(rl['gamma'], 0.9, 0.9999)
    if 'gae_lambda' in rl: rl['gae_lambda'] = np.clip(rl['gae_lambda'], 0.9, 0.99)
    if 'clip_range' in rl: rl['clip_range'] = np.clip(rl['clip_range'], 0.1, 0.4)
    if 'learning_rate' in rl: rl['learning_rate'] = np.clip(rl['learning_rate'], 1e-6, 5e-3)
    
    return mutated_params