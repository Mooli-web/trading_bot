// trading_bot_copy/trading_env_rust/src/lib.rs (Final version with fixes for warnings)
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::Rng;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Position { Flat = 0, Long = 1, Short = -1 }

// [FIX] Add PyO3 derives to expose this struct to Python if needed, and to allow easy conversion.
#[pyclass]
#[derive(Clone, Debug)]
struct TradeLog {
    #[pyo3(get)] pnl_pips: f64,
    #[pyo3(get)] direction: String,
    #[pyo3(get)] entry_price_raw: f64,
    #[pyo3(get)] entry_price_final: f64,
    #[pyo3(get)] exit_price_final: f64,
    #[pyo3(get)] stop_loss_price: f64,
    #[pyo3(get)] take_profit_price: f64,
    #[pyo3(get)] slippage_pips: f64,
    #[pyo3(get)] commission_pips: f64,
    #[pyo3(get)] duration_candles: usize,
    #[pyo3(get)] exit_reason: String,
}

const AGENT_STATE_SIZE: usize = 3;
const DEFAULT_SPREAD_PIPS: f32 = 1.0;
const PIP_SIZE: f64 = 0.0001;
const CANDLES_PER_DAY: f64 = 24.0; // Assuming H1 timeframe

#[pyclass(name = "TradingEnv")]
struct RustTradingEnv {
    #[pyo3(get)] pub action_space: PyObject,
    #[pyo3(get)] pub current_step: usize,
    #[pyo3(get)] pub price_window_size: usize,
    #[pyo3(get)] pub num_features: usize,
    
    open_prices: Vec<f32>,
    high_prices: Vec<f32>,
    low_prices: Vec<f32>,
    close_prices: Vec<f32>,
    volumes: Vec<f32>,
    
    feature_data: Vec<Vec<f32>>,
    unscaled_atr: Vec<f32>,
    spread_data: Vec<f32>,
    stop_loss_atr_multiplier: f64,
    take_profit_atr_multiplier: f64,
    
    max_steps: usize,
    position: Position,
    trade_history: Vec<TradeLog>,
    entry_price: f64,
    entry_price_raw: f64,
    stop_loss: f64,
    take_profit: f64,
    trade_start_step: usize,
    risk_per_trade_pips: f64,
    slippage_atr_fraction: f64,
    commission_pips: f64,
    swap_cost: f64,

    total_trades: usize,
    winning_trades: usize,
    total_pnl_pips: f64,
    long_trades: usize, // <--- ویژگی جدید
    short_trades: usize, // <--- ویژگی جدید
}

#[pymethods]
impl RustTradingEnv {
    #[new]
    #[pyo3(signature = ( open_py, high_py, low_py, close_py, volume_py, feature_data_py, unscaled_atr_py, agent_hyperparams, price_window_size, spread_data_py = None ))]
    fn new( py: Python, open_py: PyReadonlyArray1<f32>, high_py: PyReadonlyArray1<f32>, low_py: PyReadonlyArray1<f32>, close_py: PyReadonlyArray1<f32>, volume_py: PyReadonlyArray1<f32>, feature_data_py: PyReadonlyArray2<f32>, unscaled_atr_py: PyReadonlyArray1<f32>, agent_hyperparams: HashMap<String, PyObject>, price_window_size: usize, spread_data_py: Option<PyReadonlyArray1<f32>> ) -> PyResult<Self> {
        let spaces = py.import_bound("gymnasium.spaces")?;
        let action_space: PyObject = spaces.getattr("Discrete")?.call1((4,))?.into();

        let num_indicator_features = feature_data_py.shape()[1];
        let num_features = num_indicator_features + AGENT_STATE_SIZE;
        
        let open_prices = open_py.as_slice()?.to_vec();
        let high_prices = high_py.as_slice()?.to_vec();
        let low_prices = low_py.as_slice()?.to_vec();
        let close_prices = close_py.as_slice()?.to_vec();
        let volumes = volume_py.as_slice()?.to_vec();

        let feature_data: Vec<Vec<f32>> = feature_data_py.as_array().outer_iter().map(|row| row.to_vec()).collect();
        let unscaled_atr = unscaled_atr_py.as_slice()?.to_vec();
        let max_steps = close_prices.len() - 1;
        let spread_data = match spread_data_py {
            Some(data) => data.as_slice()?.to_vec(),
            None => vec![(DEFAULT_SPREAD_PIPS as f64 * PIP_SIZE) as f32; close_prices.len()],
        };

        let stop_loss_atr_multiplier: f64 = agent_hyperparams.get("stop_loss_atr_multiplier").and_then(|v| v.extract(py).ok()).unwrap_or(3.0);
        let take_profit_atr_multiplier: f64 = agent_hyperparams.get("take_profit_atr_multiplier").and_then(|v| v.extract(py).ok()).unwrap_or(2.5);
        let slippage_atr_fraction: f64 = agent_hyperparams.get("slippage_atr_fraction").and_then(|v| v.extract(py).ok()).unwrap_or(0.0);
        let commission_pips: f64 = agent_hyperparams.get("commission_pips").and_then(|v| v.extract(py).ok()).unwrap_or(0.0);
        let swap_pips_per_day: f64 = agent_hyperparams.get("swap_pips_per_day").and_then(|v| v.extract(py).ok()).unwrap_or(0.0);
        let swap_cost = swap_pips_per_day * PIP_SIZE;

        Ok(RustTradingEnv {
            action_space, open_prices, high_prices, low_prices, close_prices, volumes, feature_data, unscaled_atr, spread_data,
            current_step: price_window_size + 1, max_steps, position: Position::Flat,
            trade_history: Vec::new(), price_window_size, stop_loss_atr_multiplier, take_profit_atr_multiplier,
            entry_price: 0.0, entry_price_raw: 0.0, stop_loss: 0.0, take_profit: 0.0, trade_start_step: 0, risk_per_trade_pips: 0.0,
            slippage_atr_fraction, commission_pips, swap_cost, num_features,
            total_trades: 0, winning_trades: 0, total_pnl_pips: 0.0,
            long_trades: 0, short_trades: 0, // <--- مقداردهی اولیه
        })
    }

    #[pyo3(signature = (start_step = None))]
    pub fn reset<'py>(&mut self, py: Python<'py>, start_step: Option<usize>) -> PyResult<(PyObject, PyObject)> {
        if let Some(step) = start_step {
            self.current_step = step.max(self.price_window_size + 1);
        } else {
            let end_range = self.max_steps.saturating_sub(101); 
            let start_range = (self.price_window_size + 1).min(end_range);
            self.current_step = rand::thread_rng().gen_range(start_range..=end_range);
        }
        self._reset_trade_state();
        self.trade_history.clear(); // Clear history for the new episode
        self.total_trades = 0;
        self.winning_trades = 0;
        self.total_pnl_pips = 0.0;
        self.long_trades = 0; // <--- ریست کردن
        self.short_trades = 0; // <--- ریست کردن
        Ok((self._get_observation(py)?, PyDict::new_bound(py).into()))
    }

    #[pyo3(signature = (action))]
    pub fn step<'py>(&mut self, py: Python<'py>, action: u8) -> PyResult<(PyObject, f64, bool, bool, PyObject)> {
        let info = PyDict::new_bound(py);
        let mut reward = 0.0;
        
        if self.position != Position::Flat {
            let current_high = self.high_prices[self.current_step] as f64;
            let current_low = self.low_prices[self.current_step] as f64;
            let mut exit_price: Option<f64> = None;
            let mut exit_reason = "";

            let sl_hit = if self.position == Position::Long { current_low <= self.stop_loss } else { current_high >= self.stop_loss };
            let tp_hit = if self.position == Position::Long { current_high >= self.take_profit } else { current_low <= self.take_profit };

            if sl_hit { exit_price = Some(self.stop_loss); exit_reason = "stop_loss"; } 
            else if tp_hit { exit_price = Some(self.take_profit); exit_reason = "take_profit"; }
            
            if let Some(price) = exit_price {
                let trade_details = self._close_position(price, exit_reason.to_string());
                reward = self._calculate_reward(trade_details.pnl_pips);
                info.set_item("trade_closed", true)?;
                info.set_item("pnl_pips", trade_details.pnl_pips)?;
                self._reset_trade_state();
            }
        }
        
        if reward == 0.0 {
            match (self.position, action) {
                (Position::Flat, 1) => { self._open_position(Position::Long); },
                (Position::Flat, 2) => { self._open_position(Position::Short); },
                (Position::Long, 3) | (Position::Short, 3) => {
                    let pnl = self._close_position(self.close_prices[self.current_step] as f64, "agent_close".to_string()).pnl_pips;
                    reward = self._calculate_reward(pnl);
                    info.set_item("trade_closed", true)?;
                    info.set_item("pnl_pips", pnl)?;
                    self._reset_trade_state();
                },
                (Position::Long, 1) | (Position::Long, 2) | (Position::Short, 1) | (Position::Short, 2) | (Position::Flat, 3) => {
                    reward -= 0.5;
                },
                _ => {}
            }
        }
        
        self.current_step += 1;
        let is_last_step = self.current_step >= self.max_steps - 1;
        let terminated = is_last_step;
        let truncated = is_last_step;
        let obs = self._get_observation(py)?;

        if terminated || truncated {
            if self.position != Position::Flat {
                self._close_position(self.close_prices[self.current_step-1] as f64, "end_of_episode".to_string());
                self._reset_trade_state();
            }
            info.set_item("final_info", self.get_final_stats(py)?)?;
        }

        info.set_item("total_trades", self.total_trades)?;
        info.set_item("win_rate", if self.total_trades > 0 { self.winning_trades as f64 / self.total_trades as f64 } else { 0.0 })?;
        info.set_item("total_pnl_pips", self.total_pnl_pips)?;
        info.set_item("long_trades", self.long_trades)?; // <--- ارسال برای پایتون
        info.set_item("short_trades", self.short_trades)?; // <--- ارسال برای پایتون

        Ok((obs, reward, terminated, truncated, info.into()))
    }
    
    pub fn get_final_stats<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let stats = PyDict::new_bound(py);
        let pnl_history: Vec<f64> = self.trade_history.iter().map(|t| t.pnl_pips).collect();
        
        // [FIX] Return the full trade log to make use of all fields, preventing dead_code warnings.
        let trade_log_dicts: Vec<PyObject> = self.trade_history.iter().map(|log| {
            let dict = PyDict::new_bound(py);
            dict.set_item("pnl_pips", log.pnl_pips).unwrap();
            dict.set_item("direction", log.direction.clone()).unwrap();
            dict.set_item("entry_price_raw", log.entry_price_raw).unwrap();
            dict.set_item("exit_price_final", log.exit_price_final).unwrap();
            dict.set_item("duration_candles", log.duration_candles).unwrap();
            dict.set_item("exit_reason", log.exit_reason.clone()).unwrap();
            dict.into()
        }).collect();

        stats.set_item("pnl_history", PyList::new_bound(py, &pnl_history))?;
        stats.set_item("trade_history", PyList::new_bound(py, &trade_log_dicts))?;
        Ok(stats.into())
    }
}

impl RustTradingEnv {
    fn _reset_trade_state(&mut self) {
        self.position = Position::Flat;
        self.entry_price = 0.0;
        self.entry_price_raw = 0.0;
        self.stop_loss = 0.0;
        self.take_profit = 0.0;
        self.trade_start_step = 0;
        self.risk_per_trade_pips = 0.0;
    }
    
    fn _calculate_reward(&self, pnl_pips: f64) -> f64 {
        if self.risk_per_trade_pips <= 1e-9 { return 0.0; }
        let r_multiple = pnl_pips / self.risk_per_trade_pips;
        r_multiple.max(-5.0).min(5.0)
    }

    fn _apply_slippage(&self, price: f64, atr: f64, is_opening: bool, direction: Position) -> (f64, f64) {
        if self.slippage_atr_fraction <= 0.0 { return (price, 0.0); }
        let slippage_amount_price = self.slippage_atr_fraction * atr * rand::thread_rng().gen_range(0.0..=1.0);
        let final_price = price + if (is_opening && direction == Position::Long) || (!is_opening && direction == Position::Short) {
            slippage_amount_price.abs()
        } else {
            -slippage_amount_price.abs()
        };
        (final_price, slippage_amount_price / PIP_SIZE)
    }

    fn _open_position(&mut self, direction: Position) {
        self.position = direction;
        self.trade_start_step = self.current_step;
        let current_price = self.close_prices[self.current_step] as f64;
        let current_atr = self.unscaled_atr[self.current_step] as f64;
        let spread = self.spread_data[self.current_step] as f64;
        let dir_mult = if direction == Position::Long { 1.0 } else { -1.0 };
        
        self.entry_price_raw = current_price;
        let entry_price_with_spread = current_price + (spread / 2.0 * dir_mult);
        let (final_entry_price, _) = self._apply_slippage(entry_price_with_spread, current_atr, true, direction);

        self.entry_price = final_entry_price;
        self.stop_loss = self.entry_price - (current_atr * self.stop_loss_atr_multiplier * dir_mult);
        self.take_profit = self.entry_price + (current_atr * self.take_profit_atr_multiplier * dir_mult);
        self.risk_per_trade_pips = (self.entry_price - self.stop_loss).abs() / PIP_SIZE;
    }
    
    fn _close_position(&mut self, exit_price_raw: f64, reason: String) -> TradeLog {
        let current_atr = self.unscaled_atr[self.current_step] as f64;
        let spread = self.spread_data[self.current_step] as f64;
        let dir_mult = if self.position == Position::Long { 1.0 } else { -1.0 };
        
        let exit_price_with_spread = exit_price_raw - (spread / 2.0 * dir_mult);
        let (final_exit_price, slippage_pips) = self._apply_slippage(exit_price_with_spread, current_atr, false, self.position);
        
        let duration_candles = self.current_step - self.trade_start_step;
        // [FIX] Apply swap cost based on trade duration, making the `swap_cost` field used.
        let num_days_held = (duration_candles as f64 / CANDLES_PER_DAY).floor();
        let total_swap_cost_price = self.swap_cost * num_days_held * dir_mult;

        let pnl_in_price = ((final_exit_price - self.entry_price) * dir_mult) 
                         - (self.commission_pips * PIP_SIZE) 
                         - total_swap_cost_price;

        let pnl_pips = pnl_in_price / PIP_SIZE;
        
        self.total_trades += 1;
        self.total_pnl_pips += pnl_pips;
        if pnl_pips > 0.0 { self.winning_trades += 1; }

        // --- شمارشگر خرید و فروش ---
        if self.position == Position::Long {
            self.long_trades += 1;
        } else {
            self.short_trades += 1;
        }
        // -------------------------

        let trade_log = TradeLog {
            pnl_pips, 
            direction: if self.position == Position::Long { "Long".to_string() } else { "Short".to_string() },
            entry_price_raw: self.entry_price_raw, 
            entry_price_final: self.entry_price,
            exit_price_final: final_exit_price, 
            stop_loss_price: self.stop_loss,
            take_profit_price: self.take_profit, 
            slippage_pips, 
            commission_pips: self.commission_pips,
            duration_candles, 
            exit_reason: reason,
        };
        self.trade_history.push(trade_log.clone());
        trade_log
    }

    fn _get_observation<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let end_idx = self.current_step;
        let start_idx = end_idx.saturating_sub(self.price_window_size);
        let last_close = self.close_prices[end_idx - 1];
        let divisor_price = if last_close.abs() > 1e-8 { last_close } else { 1.0 };
        
        let mut candles_flat: Vec<f32> = Vec::with_capacity(self.price_window_size * 5);
        for i in start_idx..end_idx {
            candles_flat.push((self.open_prices[i] - last_close) / divisor_price);
            candles_flat.push((self.high_prices[i] - last_close) / divisor_price);
            candles_flat.push((self.low_prices[i] - last_close) / divisor_price);
            candles_flat.push((self.close_prices[i] - last_close) / divisor_price);
            candles_flat.push(self.volumes[i]);
        }
        
        let unrealized_pnl_r = if self.position != Position::Flat && self.risk_per_trade_pips > 1e-9 {
            let pnl_pips = ((self.close_prices[end_idx-1] as f64 - self.entry_price) * (if self.position == Position::Long { 1.0 } else { -1.0 })) / PIP_SIZE;
            (pnl_pips / self.risk_per_trade_pips) as f32
        } else { 0.0 };

        let (is_in_trade, time_in_trade) = if self.position != Position::Flat { 
            (1.0, ((self.current_step - self.trade_start_step) as f32) / 240.0) 
        } else { (0.0, 0.0) };
        
        let agent_state = [is_in_trade, time_in_trade, unrealized_pnl_r];
        let mut features_vec: Vec<f32> = self.feature_data[end_idx - 1].clone();
        features_vec.extend_from_slice(&agent_state);
        
        let obs_dict = PyDict::new_bound(py);
        obs_dict.set_item("candles", candles_flat.to_pyarray_bound(py).reshape([self.price_window_size, 5])?.to_owned())?;
        obs_dict.set_item("features", features_vec.to_pyarray_bound(py).to_owned())?;
        Ok(obs_dict.into())
    }
}

#[pymodule]
fn trading_env_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustTradingEnv>()?;
    m.add_class::<TradeLog>()?; // Expose the TradeLog to Python
    Ok(())
}