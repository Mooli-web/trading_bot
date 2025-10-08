# main.py (Final version, OS-agnostic)
import os
import argparse
import pandas as pd
import logging
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sb3_contrib import RecurrentPPO
import time
import multiprocessing as mp

# Rich library is now used for terminal UI
from rich.logging import RichHandler

from src.indicator_calculator import IndicatorCalculator
from src.evolutionary_trainer import EvolutionaryTrainer
from src.evaluation import evaluate_signals
from config.settings import DataSplit, Training, WalkForward
from src.live_monitor import LiveMonitor

def setup_logging():
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("joblib").setLevel(logging.WARNING) # Quieten joblib logs

logger = logging.getLogger(__name__)

def load_raw_data(file_path: str, timeframe: str) -> pd.DataFrame:
    logger.info(f"--- 1. Loading Raw {timeframe} Data ---")
    if not os.path.exists(file_path):
        logger.critical(f"Data file not found. Path: '{file_path}'")
        raise FileNotFoundError(f"{timeframe} data file is missing.")
    data = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    logger.info(f"Loaded {len(data)} rows of {timeframe} data.")
    return data

def get_resume_state(state_file_path: str) -> dict | None:
    if os.path.exists(state_file_path):
        try:
            state = joblib.load(state_file_path)
            logger.info(f"Successfully loaded state file. Resuming from Fold {state.get('fold', 1)}.")
            return state
        except Exception as e:
            logger.error(f"Could not load state file at '{state_file_path}'. Starting fresh. Error: {e}")
            return None
    return None

def run_walk_forward_training(h1_df_raw: pd.DataFrame, daily_df_raw: pd.DataFrame, args: argparse.Namespace):
    logger.info("--- 2. Preparing Data Splits for Walk-Forward Validation ---")
    
    indicator_calculator = IndicatorCalculator()
    feature_columns = indicator_calculator.get_feature_columns()

    start_fold = 1
    resume_state = get_resume_state(Training.STATE_FILE_PATH) if args.resume else None
    if resume_state:
        start_fold = resume_state.get('fold', 1)

    n_samples = len(h1_df_raw)
    test_set_size = int(n_samples * DataSplit.TEST_RATIO) if DataSplit.TEST_RATIO > 0 else 0
    
    walk_forward_data_raw = h1_df_raw.iloc[:-test_set_size] if test_set_size > 0 else h1_df_raw
    final_test_df_raw = h1_df_raw.iloc[-test_set_size:] if test_set_size > 0 else pd.DataFrame()

    n_wf_samples = len(walk_forward_data_raw)
    initial_train_size = int(n_wf_samples * WalkForward.INITIAL_TRAIN_RATIO)
    remaining_for_validation = n_wf_samples - initial_train_size
    validation_step_size = remaining_for_validation // WalkForward.NUM_SPLITS

    logger.info(f"Total raw data points: {n_samples}")
    logger.info(f"Walk-Forward raw data: {n_wf_samples} rows")
    logger.info(f"Final Hold-out raw data: {len(final_test_df_raw)} rows")
    logger.info(f"Total walk-forward splits (folds): {WalkForward.NUM_SPLITS}")
    
    try:
        with open(args.hyperparams_file, 'r', encoding='utf-8') as f:
            base_hyperparams = {'param_ranges': json.load(f).get('rl_params', {})}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.critical(f"Could not load or parse hyperparameters file '{args.hyperparams_file}'. Error: {e}")
        return

    elites_from_previous_fold = None
    if resume_state and start_fold > 1 and resume_state.get('best_agent_overall'):
        elites_from_previous_fold = [resume_state['best_agent_overall']]

    best_agent_from_last_fold = None

    for i in range(start_fold - 1, WalkForward.NUM_SPLITS):
        fold_num = i + 1
        train_end_idx = initial_train_size + i * validation_step_size
        validation_end_idx = train_end_idx + validation_step_size
        if fold_num == WalkForward.NUM_SPLITS: validation_end_idx = n_wf_samples

        train_df_raw = walk_forward_data_raw.iloc[:train_end_idx].copy()
        validation_df_raw = walk_forward_data_raw.iloc[train_end_idx:validation_end_idx].copy()
        
        logger.info(f"\nProcessing training data ({len(train_df_raw)} rows)...")
        train_df = indicator_calculator.add_indicators(train_df_raw, daily_df_raw)
        
        logger.info(f"Processing validation data ({len(validation_df_raw)} rows)...")
        validation_df = indicator_calculator.add_indicators(validation_df_raw, daily_df_raw)
        
        if train_df.empty or validation_df.empty:
            logger.critical(f"Fold {fold_num}: DataFrame became empty after indicator calculation.")
            return

        scaler = StandardScaler()
        train_df.loc[:, feature_columns] = scaler.fit_transform(train_df[feature_columns])
        validation_df.loc[:, feature_columns] = scaler.transform(validation_df[feature_columns])
        
        if fold_num == WalkForward.NUM_SPLITS:
            joblib.dump(scaler, Training.SCALER_PATH)
            logger.info(f"Scaler from final fold saved to '{Training.SCALER_PATH}'.")

        with LiveMonitor(fold_num, WalkForward.NUM_SPLITS) as monitor:
            trainer = EvolutionaryTrainer(
                train_df=train_df, validation_df=validation_df, indicator_calculator=indicator_calculator,
                base_hyperparams=base_hyperparams, resume=args.resume, fold_number=fold_num,
                initial_population=elites_from_previous_fold, monitor=monitor
            )
            trainer.run()
            elites_from_previous_fold = trainer.get_elites()
            if fold_num == WalkForward.NUM_SPLITS:
                best_agent_from_last_fold = trainer.best_agent_overall

    run_final_evaluation(final_test_df_raw, daily_df_raw, best_agent_from_last_fold, indicator_calculator)

def run_final_evaluation(test_df_raw: pd.DataFrame, daily_df_raw: pd.DataFrame, best_agent, indicator_calculator):
    final_model_path = os.path.join(Training.MODEL_SAVE_PATH, "final_model.zip")
    if test_df_raw.empty or best_agent is None or not os.path.exists(final_model_path):
        logger.warning("Skipping final hold-out test.")
        return
        
    logger.info(f"\n{'='*25} FINAL EVALUATION ON HOLD-OUT TEST SET {'='*25}")
    final_model = RecurrentPPO.load(final_model_path)
    scaler = joblib.load(Training.SCALER_PATH)
    
    test_df = indicator_calculator.add_indicators(test_df_raw, daily_df_raw)
    test_df.loc[:, indicator_calculator.get_feature_columns()] = scaler.transform(test_df[indicator_calculator.get_feature_columns()])
    
    logger.info("Starting final signal evaluation...")
    evaluate_signals(
        model=final_model, eval_data=test_df, indicator_calculator=indicator_calculator,
        agent_hyperparams=best_agent.hyperparams['rl_params'], agent_id=f"FINAL_TEST_{best_agent.id}"
    )
    logger.info("Final hold-out signal evaluation complete.")

def main(args):
    os.makedirs(Training.MODEL_SAVE_PATH, exist_ok=True)
    logger.info("Verified that 'models' directory exists.")

    try:
        data_path = os.path.join('data', 'raw')
        h1_df_raw = load_raw_data(os.path.join(data_path, args.data_file), "H1")
        daily_df_raw = load_raw_data(os.path.join(data_path, args.daily_data_file), "Daily")
        run_walk_forward_training(h1_df_raw, daily_df_raw, args)

    except FileNotFoundError as e:
        logger.critical(f"Execution stopped: {e}")
    except Exception:
        logger.critical("An unexpected error occurred.", exc_info=True)

if __name__ == '__main__':
    # [FIX] The fork method is not available on Windows.
    # We remove the attempt to set it and will rely on joblib's default
    # robust backend ('loky') which is designed for cross-platform compatibility.
    
    parser = argparse.ArgumentParser(description="Evolutionary Signal Generation Bot Training Pipeline")
    parser.add_argument('--data-file', type=str, default='EURUSD_H1.csv', help='Name of the H1 data file in data/raw/.')
    parser.add_argument('--daily-data-file', type=str, default='EURUSD_D1.csv', help='Name of the Daily data file in data/raw/.')
    parser.add_argument('--hyperparams-file', type=str, default='config/hyperparameters.json', help='The JSON file for hyperparameter ranges.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last saved state.')
    
    args = parser.parse_args()
    setup_logging()
    
    logger.info("--- SIGNAL GENERATION PIPELINE STARTED ---")
    logger.info(f"H1 Data: {args.data_file}, Daily Data: {args.daily_data_file}")
    logger.info(f"Hyperparameters: {args.hyperparams_file}, Resume Training: {args.resume}")
    logger.info("------------------------------------")
    
    main(args)
    logger.info("--- SIGNAL GENERATION PIPELINE FINISHED ---")