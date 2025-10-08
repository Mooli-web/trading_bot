# src/callbacks.py (Final version - Simplified and robust)
import numpy as np
import logging
from stable_baselines3.common.callbacks import BaseCallback
import multiprocessing as mp
from config.settings import Training

logger = logging.getLogger(__name__)

class TradingMetricsCallback(BaseCallback):
    def __init__(self, agent_id: str, update_queue: mp.Queue, verbose: int = 0):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.update_queue = update_queue
        # [FIX] Send an update less frequently to reduce overhead
        self.log_freq = 250 # <--- تغییر فرکانس آپدیت

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        infos = self.locals.get("infos", [{}])
        if infos:
            info = infos[0]
            action = self.locals['actions'][0]
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}

            update_data = {
                'id': self.agent_id,
                'status': 'Training',
                'step': f"{self.num_timesteps}/{Training.TOTAL_TIMESTEPS_PER_AGENT}",
                'trades': info.get('total_trades', 0),
                'win_rate': info.get('win_rate', 0.0) * 100,
                'pnl': info.get('total_pnl_pips', 0.0),
                'action': action_map.get(action, 'N/A'),
                'buys': info.get('long_trades', 0), # <--- ارسال تعداد خرید
                'sells': info.get('short_trades', 0), # <--- ارسال تعداد فروش
            }
            try:
                self.update_queue.put_nowait(update_data)
            except Exception:
                pass

        return True

    # Note: The _on_training_end is removed because the `finally` block
    # in `evolutionary_trainer.py` is a more reliable way to signal completion,
    # as it catches exceptions during initialization as well.