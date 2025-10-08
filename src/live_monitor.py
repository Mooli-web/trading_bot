# src/live_monitor.py (Final version with robust queue handling)
import time
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Group
from rich.text import Text
from queue import Empty

class LiveMonitor:
    def __init__(self, fold_num: int, total_folds: int):
        self.fold_num = fold_num
        self.total_folds = total_folds
        self.layout = self._make_layout()
        self.agent_status = {}
        self.live = Live(self.layout, screen=True, redirect_stderr=False, refresh_per_second=10)

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.live.stop()

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(ratio=1, name="main"),
            Layout(size=5, name="footer")
        )
        layout["main"].split_row(Layout(name="side"), Layout(name="body", ratio=2))
        layout["side"].split(Layout(name="progress"), Layout(name="summary"))
        return layout

    def start_generation(self, gen_num: int, total_gens: int, population: list):
        self.gen_num = gen_num
        self.total_gens = total_gens
        self.agent_status = {
            agent.id: {
                "status": "Waiting", "step": "0", "trades": 0, "win_rate": 0,
                "pnl": 0.0, "action": "-", "buys": 0, "sells": 0,
                "duration": 0.0, "sharpe": 0.0, "drawdown": 0.0,
                "final_fitness": 0.0
            }
            for agent in population
        }
        self._update_header()
        self._update_progress()
        self._update_agent_table()

    def _update_header(self):
        header_text = Text(f"Walk-Forward Training: Fold {self.fold_num}/{self.total_folds}", justify="center", style="bold magenta")
        self.layout["header"].update(Panel(header_text))

    def _update_progress(self):
        gen_progress = Progress(TextColumn("[bold blue]Generations"), BarColumn(), TextColumn("{task.completed}/{task.total}"))
        gen_task = gen_progress.add_task("Gen", total=self.total_gens, completed=self.gen_num - 1)
        self.layout["progress"].update(Panel(gen_progress, title="Overall Progress"))

    def _update_agent_table(self):
        table = Table(title=f"Agent Status - Generation {self.gen_num}")
        table.add_column("Agent ID", justify="left", style="cyan", no_wrap=True)
        # --- [FIX] افزایش عرض ستون Status ---
        table.add_column("Status", justify="left", min_width=45)
        table.add_column("Duration", justify="right", style="dim")
        table.add_column("Timestep", justify="right", style="green")
        table.add_column("Buys", justify="right", style="green")
        table.add_column("Sells", justify="right", style="red")
        table.add_column("Win Rate", justify="right", style="yellow")
        table.add_column("Train PnL", justify="right")
        table.add_column("Eval Fitness", justify="right", style="bold")

        status_style_map = {
            "Training": "[yellow]Training[/]", "Waiting": "[dim]Waiting[/]",
            "Initializing": "[cyan]Initializing[/]", "Evaluating": "[blue]Evaluating[/]",
        }

        for agent_id, data in sorted(self.agent_status.items()):
            status = data.get('status', 'N/A')
            train_pnl = data.get('pnl', 0.0)
            final_fitness = data.get('final_fitness', 0.0)

            pnl_style = "green" if train_pnl > 0 else "red" if train_pnl < 0 else "white"
            fitness_style = "bold green" if final_fitness > 0 else "bold red" if final_fitness < 0 else "white"

            duration_s = data.get('duration', 0)
            duration_str = f"{int(duration_s // 60)}m {int(duration_s % 60)}s" if duration_s > 0 else "-"

            if status == 'Done':
                # --- [FIX] نمایش تمام اطلاعات در یک خط ---
                pf = data.get('profit_factor', 0.0)
                sharpe = data.get('sharpe', 0.0)
                drawdown = data.get('drawdown', 0.0)
                avg_dur = data.get('avg_duration', 0.0)
                metrics_str = f"PF:{pf:.2f} S:{sharpe:.2f} DD:{drawdown:.1f} Dur:{avg_dur:.1f}h"
                display_status = Text("Done | ", style="bold green") + Text(metrics_str, style="dim")
            else:
                display_status = status_style_map.get(status, status)

            table.add_row(
                agent_id,
                display_status,
                duration_str,
                str(data['step']),
                str(data.get('buys', 0)),
                str(data.get('sells', 0)),
                f"{data['win_rate']:.1f}%",
                Text(f"{train_pnl:.2f}", style=pnl_style),
                Text(f"{final_fitness:.2f}", style=fitness_style) if status == 'Done' else "-"
            )
        self.layout["body"].update(Panel(table))

    def process_updates(self, queue, total_agents):
        finished_agents = 0
        while finished_agents < total_agents:
            try:
                update = queue.get(timeout=1.0)
                agent_id = update.get('id')
                if not agent_id: continue

                current_agent_status = self.agent_status.get(agent_id, {})
                current_agent_status.update(update)
                self.agent_status[agent_id] = current_agent_status

                if update.get('status') == 'Done':
                    finished_agents += 1
                    final_metrics = update.get('final_metrics', {})
                    current_agent_status.update(final_metrics)
                    current_agent_status['final_fitness'] = update.get('fitness', 0)

            except Empty:
                pass

            self._update_agent_table()

    def log_generation_summary(self, best_agent, avg_fitness: float, custom_message: str = ""):
        if custom_message:
            summary_text = Text(custom_message, style="bold yellow")
        else:
            summary_text = Group(
                Text(f"Best Agent: [bold cyan]{best_agent.id}[/] (Fitness: [bold green]{best_agent.fitness:.2f}[/])"),
                Text(f"Avg Fitness: [yellow]{avg_fitness:.2f}[/]")
            )
        self.layout["footer"].update(Panel(summary_text, title="Generation Summary"))
        self.live.refresh()