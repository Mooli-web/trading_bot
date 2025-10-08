# src/indicator_calculator.py (Final version with Multi-Timeframe and Time Features)
import pandas as pd
import numpy as np
import logging
import pandas_ta as ta

from config.settings import Indicators

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    def __init__(self):
        self.rsi_period = Indicators.RSI_PERIOD
        self.adx_period = Indicators.ADX_PERIOD
        self.atr_period = Indicators.ATR_PERIOD
        self.bbands_period = Indicators.BBANDS_PERIOD
        self.donchian_period = Indicators.DONCHIAN_PERIOD
        logger.debug("IndicatorCalculator initialized with advanced feature engineering logic.")

    def add_indicators(self, data: pd.DataFrame, daily_data: pd.DataFrame = None) -> pd.DataFrame:
        if data.empty:
            logger.warning("Input dataframe is empty. No operations performed.")
            return data

        df = data.copy()
        logger.info(f"Starting feature calculation for a dataframe with {len(df)} rows...")

        df.columns = [col.lower() for col in df.columns]
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                logger.critical(f"Required column '{col}' not found in dataframe. Halting operations.")
                raise ValueError(f"Required column '{col}' not found. Please check your CSV file.")
        
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_rows = len(df)
        df.dropna(subset=required_cols, inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"{initial_rows - len(df)} rows were dropped due to invalid OHLC values.")
        
        if df.empty:
            logger.error("Dataframe became empty after initial cleaning. Input data quality is very low.")
            return df

        # --- Base Indicators ---
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        df['RSI'] = ta.rsi(df['close'], length=self.rsi_period)
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
        if adx_df is not None and not adx_df.empty:
            adx_col_name = f'ADX_{self.adx_period}'
            if adx_col_name in adx_df.columns:
                df['ADX'] = adx_df[adx_col_name]
            else:
                df['ADX'] = np.nan
        
        bbands_df = ta.bbands(df['close'], length=self.bbands_period)
        if bbands_df is not None and not bbands_df.empty and bbands_df.shape[1] >= 3:
            df['BB_lower'] = bbands_df.iloc[:, 0]
            df['BB_mid'] = bbands_df.iloc[:, 1]
            df['BB_upper'] = bbands_df.iloc[:, 2]
        else:
            df['BB_lower'], df['BB_mid'], df['BB_upper'] = np.nan, np.nan, np.nan
        
        donchian_df = ta.donchian(df['high'], df['low'], length=self.donchian_period)
        if donchian_df is not None and not donchian_df.empty and donchian_df.shape[1] >= 2:
            df['donchian_lower'] = donchian_df.iloc[:, 0]
            df['donchian_upper'] = donchian_df.iloc[:, 1]
        else:
            df['donchian_lower'], df['donchian_upper'] = np.nan, np.nan

        # --- Engineered Features ---
        df['BB_width_normalized'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
        df['donchian_pos'] = (df['close'] - df['donchian_lower']) / (df['donchian_upper'] - df['donchian_lower'])
        df['donchian_pos'] = df['donchian_pos'].clip(0, 1)

        df['body_size_norm'] = (df['close'] - df['open']).abs() / df['ATR']
        df['upper_wick_norm'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['ATR']
        df['lower_wick_norm'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['ATR']
        
        df['volatility_6h'] = df['close'].pct_change().rolling(window=6).std() * 100

        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        if daily_data is not None and not daily_data.empty:
            daily_data_c = daily_data.copy()
            daily_data_c.columns = [col.lower() for col in daily_data_c.columns]
            daily_sma = ta.sma(daily_data_c['close'], length=50)
            daily_sma.name = "daily_sma_50"
            
            df = pd.merge(df, daily_sma, left_index=True, right_index=True, how='left')
            
            # <<-- تغییر کلیدی: استفاده از روش جدید و امن برای پر کردن مقادیر خالی -->>
            df['daily_sma_50'] = df['daily_sma_50'].ffill()
            
            df['dist_from_daily_sma_norm'] = (df['close'] - df['daily_sma_50']) / df['ATR']
        
        feature_cols = self.get_feature_columns()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        required_env_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'ATR']
        final_df = df[required_env_cols + feature_cols].copy()

        initial_rows = len(final_df)
        final_df.dropna(inplace=True)
        final_rows = len(final_df)
        
        if initial_rows > final_rows:
            logger.info(f"{initial_rows - final_rows} initial rows were dropped due to indicator warm-up period.")

        if final_df.empty:
            logger.critical("Dataframe is empty after all calculations. Input data might be too short for the indicator periods.")
        else:
            logger.info(f"Feature calculation successful. Final data points: {len(final_df)}")
        
        final_df[feature_cols] = final_df[feature_cols].astype('float64')
        return final_df

    def get_feature_columns(self) -> list[str]:
        return [
            'RSI', 'ADX', 'BB_width_normalized', 'donchian_pos', 
            'body_size_norm', 'upper_wick_norm', 'lower_wick_norm',
            'volatility_6h', 'hour_of_day', 'day_of_week',
            'dist_from_daily_sma_norm'
        ]