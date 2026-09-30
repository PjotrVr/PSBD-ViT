"""The Telemetry the training loop calls, and the lines it writes.

Nothing here computes a statistic. It decides when each probe runs, holds what a
probe keeps between windows and writes <checkpoint dir>/telemetry.jsonl: 1 header
line, 1 window line every TelemetryConfig.every steps (heavy readings riding on
the lines whose step closes a heavy window) and 1 epoch line per epoch.

The run it observes trains exactly as it would without it. The light probe reads
logits the update already computed, the heavy and epoch probes run in eval mode
under no_grad and restore the training mode, and every probe that draws random
numbers (the token masks) runs inside a forked RNG, so the data order, the model
and every later draw are the ones a run without telemetry would see.
"""

import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from utils.provenance import utc_timestamp
from .dormancy import hidden_layer_readings
from .heldout import HeldoutPairs, epoch_readings
from .modules import UpdateObserver, module_keys
from .rows import ROW_GROUPS, row_group_codes, row_statistic_keys, row_statistic_sums
from .window import (
    WindowAccumulators,
    append_record,
    rounded,
    start_record_file,
)

TELEMETRY_FILENAME = "telemetry.jsonl"


@dataclass(frozen=True)
class TelemetryConfig:
    """What the telemetry measures and how often.

    every is the light window in optimizer steps. heavy_every is the heavy cadence,
    0 to switch the heavy probes off, and must be a multiple of every so that a
    heavy reading always lands on a line. retention_rates are the PSBD-TM rates the
    epoch record reads, empty to skip retention. measure_surplus gates the surplus
    factor. The heavy dormancy readings and every epoch probe need held-out pairs,
    and without them only the light window and the module norms are written.
    """

    every: int = 25
    heavy_every: int = 100
    retention_rates: tuple[float, ...] = (0.5, 0.8)
    measure_surplus: bool = True

    def __post_init__(self):
        if self.every <= 0:
            raise ValueError(f"telemetry every must be positive, got {self.every}")
        if self.heavy_every < 0 or self.heavy_every % self.every:
            raise ValueError(
                f"telemetry heavy_every {self.heavy_every} must be 0 or a multiple "
                f"of every {self.every}, so a heavy reading lands on a window line"
            )


def forked_rng(device: torch.device):
    """A context restoring the CPU generator, and the device's when it is CUDA."""
    devices = [device] if device.type == "cuda" else []
    context = torch.random.fork_rng(devices=devices)
    return context


class Telemetry:
    """Windowed training telemetry for 1 run, written to 1 JSONL file.

    The loop calls begin_step before each update, observe_batch with the update's
    logits, end_step after it, on_epoch_end after validation and close once at
    the end.
    """

    def __init__(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        architecture: str,
        config: TelemetryConfig,
        training_dataset,
        heldout: HeldoutPairs | None,
        device: torch.device,
        use_bfloat16: bool,
    ):
        self.path = path
        self.model = model
        self.architecture = architecture
        self.config = config
        self.device = device
        self.use_bfloat16 = use_bfloat16
        self.heldout = heldout.to(device) if heldout is not None else None

        row_codes = row_group_codes(training_dataset)  # (rows,)
        self.row_codes = row_codes.to(device)
        self.window = WindowAccumulators(device)
        self.observer = (
            UpdateObserver(model, optimizer, architecture)
            if config.heavy_every > 0
            else None
        )

        self.step = 0
        self.epoch = 1
        self._window_opened_at = time.perf_counter()
        self._window_opened_step = 0
        self._window_probe_seconds = 0.0
        self._heavy_count = 0

        header = {
            "record": "header",
            "started_at": utc_timestamp(),
            "architecture": architecture,
            "config": asdict(config),
            "row_counts": {
                group: int((row_codes == code).sum())
                for code, group in enumerate(ROW_GROUPS)
            },
            "heldout_pairs": int(heldout.clean.shape[0]) if heldout is not None else 0,
            "module_groups": self.observer.group_names if self.observer else [],
        }
        start_record_file(path, header)

    def heavy_due(self, step: int) -> bool:
        due = self.config.heavy_every > 0 and step % self.config.heavy_every == 0
        return due

    def begin_step(self) -> None:
        """Advance the step count and arm the optimizer hooks on a heavy step."""
        self.step += 1
        if self.observer is not None and self.heavy_due(self.step):
            self.observer.arm()

    def observe_batch(
        self, logits: torch.Tensor, labels: torch.Tensor, row_indices: torch.Tensor
    ) -> None:
        """Accumulate the light split from the update's (batch, classes) logits."""
        started = time.perf_counter()
        groups = self.row_codes[row_indices.to(self.device)]  # (batch,)
        self.window.add("rows", row_statistic_sums(logits, labels, groups))
        self._window_probe_seconds += time.perf_counter() - started

    def end_step(self) -> None:
        """Take the heavy readings on a heavy step, and flush when a window closes."""
        started = time.perf_counter()
        if self.heavy_due(self.step):
            self._take_heavy_readings()
        self._window_probe_seconds += time.perf_counter() - started

        if self.step % self.config.every == 0:
            self.flush()

    def _take_heavy_readings(self) -> None:
        self._heavy_count += 1
        reading = self.observer.take() if self.observer is not None else None
        if reading is not None:
            self.window.add("modules", reading)

        if self.heldout is None:
            return
        was_training = self.model.training
        with forked_rng(self.device):
            for group, images in (
                ("clean", self.heldout.clean),
                ("triggered", self.heldout.triggered),
            ):
                readings = hidden_layer_readings(
                    self.model,
                    self.architecture,
                    images,
                    self.device,
                    self.use_bfloat16,
                )
                for name, value in readings.items():
                    self.window.add(f"hidden/{name}_{group}", value)
        self.model.train(was_training)

    def flush(self) -> None:
        """Close the window: 1 transfer off the device, 1 line, then reset."""
        if self.window.is_empty:
            return

        closed_at = time.perf_counter()
        sums = self.window.read()
        record = {
            "record": "window",
            "step": self.step,
            "epoch": self.epoch,
            "window_steps": self.step - self._window_opened_step,
            "window_seconds": round(closed_at - self._window_opened_at, 3),
            "telemetry_seconds": round(self._window_probe_seconds, 4),
            "heavy": self._heavy_count > 0,
        }
        if "rows" in sums:
            record.update(row_statistic_keys(sums["rows"]))
        if "modules" in sums and self.observer is not None:
            record.update(
                module_keys(
                    sums["modules"], self.observer.group_names, self._heavy_count
                )
            )
        for name, values in sums.items():
            if name.startswith("hidden/"):
                record[name.removeprefix("hidden/")] = rounded(
                    values[0] / self._heavy_count
                )

        append_record(self.path, record)
        self.window.clear()
        self._window_opened_at = time.perf_counter()
        self._window_opened_step = self.step
        self._window_probe_seconds = 0.0
        self._heavy_count = 0

    def on_epoch_end(
        self, epoch: int, validation_accuracy: float, training_loss: float
    ) -> None:
        """Write the epoch line: the loop's own numbers and the held-out readings."""
        started = time.perf_counter()
        record = {
            "record": "epoch",
            "epoch": epoch,
            "step": self.step,
            "training_loss": rounded(training_loss),
            "validation_accuracy": rounded(validation_accuracy),
        }
        if self.heldout is not None:
            was_training = self.model.training
            with forked_rng(self.device):
                record.update(
                    epoch_readings(
                        self.model,
                        self.architecture,
                        self.heldout,
                        self.config.retention_rates,
                        self.config.measure_surplus,
                        self.device,
                        self.use_bfloat16,
                    )
                )
            self.model.train(was_training)
        record["telemetry_seconds"] = round(time.perf_counter() - started, 4)

        append_record(self.path, record)
        self.epoch = epoch + 1

    def close(self) -> None:
        """Flush the open window and release the optimizer hooks."""
        self.flush()
        if self.observer is not None:
            self.observer.remove()
