"""The device side of a telemetry window and the single transfer that closes it.

Nothing here reads a tensor's value while a window is open. Every accumulator is a
flat float64 device tensor in 1 dictionary, so a flush is 1 cat and 1 copy to the
host whatever the run switched on. There is exactly 1 copy to the host in the
window machinery, and it is WindowAccumulators.read.
"""

import json
import math

import numpy as np
import torch


class WindowAccumulators:
    """Named running sums on the device, read back in 1 transfer."""

    def __init__(self, device: torch.device):
        self.device = device
        self._sums: dict[str, torch.Tensor] = {}

    @property
    def is_empty(self) -> bool:
        empty = not self._sums
        return empty

    def add(self, name: str, value: torch.Tensor) -> None:
        """Add value into the accumulator of that name, allocating it on first use."""
        flat = value.detach().to(torch.float64).reshape(-1)  # (width,)
        held = self._sums.get(name)
        if held is None:
            self._sums[name] = flat.clone()
            return
        held += flat

    def read(self) -> dict[str, np.ndarray]:
        """Every accumulator as a numpy array, in 1 transfer off the device."""
        if not self._sums:
            return {}

        names = list(self._sums)
        widths = [self._sums[name].numel() for name in names]
        flat = torch.cat([self._sums[name] for name in names]).cpu().numpy()  # (total,)

        arrays = {}
        offset = 0
        for name, width in zip(names, widths):
            arrays[name] = flat[offset : offset + width]
            offset += width
        return arrays

    def clear(self) -> None:
        self._sums = {}


def rounded(value, digits: int = 5) -> float | None:
    """A number as a record writes it, None where it is not finite."""
    number = float(value)
    if not math.isfinite(number):
        return None
    rounded_number = round(number, digits)
    return rounded_number


def flat_key(name: str) -> str:
    """A module path as a record key, with every dot and slash replaced by "_"."""
    flattened = name.replace(".", "_").replace("/", "_")
    return flattened


def rate_tag(rate: float) -> str:
    """A rate as a key fragment in the checkpoint-folder convention, 0.5 as 0_5."""
    tag = f"{rate:g}".replace(".", "_")
    return tag


def start_record_file(path: str, header: dict) -> None:
    """Truncate the file and write its header line.

    A rerun into the same checkpoint folder overwrites the checkpoint too, so
    appending to an older run's lines would mix 2 runs under 1 file.
    """
    with open(path, "w") as handle:
        handle.write(json.dumps(header) + "\n")


def append_record(path: str, record: dict) -> None:
    """Append 1 record as 1 JSON line."""
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")
