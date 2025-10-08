# config/settings.py (Final version with restored classes)
import os

# [FIX] Restored the missing Indicators and Agent classes
class Indicators:
    RSI_PERIOD = 14
    SMA_SHORT = 50
    SMA_LONG = 200
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    BBANDS_PERIOD = 20
    DONCHIAN_PERIOD = 20

class Agent:
    STOP_LOSS_ATR_MULTIPLIER = 3.0
    TAKE_PROFIT_ATR_MULTIPLIER = 2.5
    MAX_TRADE_DURATION = 72
    PRICE_WINDOW_SIZE = 48
    SLIPPAGE_ATR_FRACTION = 0.1
    COMMISSION_PIPS = 0.7
    SWAP_PIPS_PER_DAY = -0.2

class Training:
    MODEL_SAVE_PATH = "models/"
    SCALER_PATH = os.path.join(MODEL_SAVE_PATH, "scaler.joblib")
    STATE_FILE_PATH = os.path.join(MODEL_SAVE_PATH, "evolutionary_state.joblib")
    # --- کاهش تعداد قدم‌ها طبق استراتژی جدید ---
    TOTAL_TIMESTEPS_PER_AGENT = 50000

class DataSplit:
    TEST_RATIO = 0.15

class WalkForward:
    NUM_SPLITS = 5
    INITIAL_TRAIN_RATIO = 0.40

class Evolutionary:
    # --- تنظیمات جدید برای آموزش گسترده ---
    POPULATION_SIZE = 20
    NUM_GENERATIONS = 50
    # ------------------------------------
    ELITISM_COUNT = 2
    TOURNAMENT_SIZE = 5 # کمی افزایش برای رقابت بیشتر در جمعیت بزرگتر
    CROSSOVER_PROBABILITY = 0.7
    MUTATION_PROBABILITY_START = 0.6
    MUTATION_PROBABILITY_END = 0.1
    STRONG_MUTATION_PROBABILITY = 0.1
    N_JOBS = 4 # Use all available CPU cores
    NUM_EVAL_RUNS = 3 # Reduce for faster runs