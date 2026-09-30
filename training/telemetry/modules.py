"""Per-module gradient and parameter norms, and how far 1 optimizer step moved each module.

Ported from ek_solver's solver/dmc/telemetry/modules.py. The distance moved is
MEASURED, not reconstructed: an Adam step is not the learning rate times the
gradient, and an update ratio built from the gradient would report a step the run
never takes. The observer snapshots the parameters just before the optimizer step
and differences them just after it, through the optimizer's own step hooks, so
nothing in the update function changes.

The hooks sit on the base optimizer. Under SAM that is the Adam that second_step
calls after restoring the original weights, so the snapshot holds the real
weights, the gradient is the one taken at the perturbed point and the difference
is the sharpness-aware update itself. The gradient read is the one the step
consumed, after any --clip-grad-norm clipping.

Modules are grouped per transformer block and per block child (ViT's ln_1,
self_attention, ln_2 and mlp, Swin's norm1, attn, norm2 and mlp) and outside
the blocks by the parameter's owning module (the patch embedding, the class
token, the final norm and the head).
"""

import numpy as np
import torch
import torch.nn as nn

from analysis.features import transformer_blocks
from models.backbones import network_core
from .window import flat_key, rounded

# Rows of the (3, groups) reading, in this order.
MODULE_READINGS = ("grad_squared", "param_squared", "moved_squared")


def parameter_group_names(model: nn.Module, architecture: str) -> list[str]:
    """The group of every trainable parameter, in model.parameters() order."""
    core = network_core(model)
    blocks = transformer_blocks(core, architecture)
    block_number = {id(block): index + 1 for index, block in enumerate(blocks)}
    block_prefixes = {
        name: block_number[id(module)]
        for name, module in core.named_modules()
        if id(module) in block_number
    }

    group_of_parameter = {}
    for name, parameter in core.named_parameters():
        group_of_parameter[id(parameter)] = parameter_group(name, block_prefixes)

    names = [
        group_of_parameter[id(parameter)]
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return names


def parameter_group(name: str, block_prefixes: dict[str, int]) -> str:
    """block{NN}_{child} inside a transformer block, the owning module path outside."""
    for prefix, number in block_prefixes.items():
        if name.startswith(prefix + "."):
            child = name[len(prefix) + 1 :].split(".")[0]
            group = f"block{number:02d}_{child}"
            return group

    owner = name.rsplit(".", 1)[0] if "." in name else name
    group = flat_key(owner)
    return group


class UpdateObserver:
    """Reads gradient, parameter and moved norms around 1 armed optimizer step.

    arm() before the update, take() after it. An unarmed step costs 2 attribute
    reads in the hooks and nothing else.
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, architecture: str
    ):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        names = parameter_group_names(model, architecture)
        self.group_names = sorted(set(names))
        group_position = {name: index for index, name in enumerate(self.group_names)}
        first_device = self.parameters[0].device
        self.group_index = torch.tensor(
            [group_position[name] for name in names], device=first_device
        )  # (parameters,)

        self._armed = False
        self._before: list[torch.Tensor] | None = None
        self._reading: torch.Tensor | None = None

        stepped = getattr(optimizer, "base_optimizer", optimizer)
        self._handles = [
            stepped.register_step_pre_hook(self._before_step),
            stepped.register_step_post_hook(self._after_step),
        ]

    def arm(self) -> None:
        self._armed = True

    def take(self) -> torch.Tensor | None:
        """The last armed step's (3, groups) squared sums on the device, None if none."""
        reading = self._reading
        self._reading = None
        return reading

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @torch.no_grad()
    def _before_step(self, _optimizer, _args, _kwargs) -> None:
        if not self._armed:
            return
        self._before = [p.detach().clone() for p in self.parameters]

    @torch.no_grad()
    def _after_step(self, _optimizer, _args, _kwargs) -> None:
        if not self._armed or self._before is None:
            return

        # The gradient survives the step (zero_grad runs at the start of the next
        # update), and it is the one the step consumed.
        grad_squared = torch.stack(
            [
                p.grad.detach().float().pow(2).sum()
                if p.grad is not None
                else torch.zeros((), device=p.device)
                for p in self.parameters
            ]
        )  # (parameters,)
        param_squared = torch.stack(
            [before.float().pow(2).sum() for before in self._before]
        )  # (parameters,)
        moved_squared = torch.stack(
            [
                (p.detach().float() - before.float()).pow(2).sum()
                for p, before in zip(self.parameters, self._before)
            ]
        )  # (parameters,)

        per_parameter = torch.stack(
            [grad_squared, param_squared, moved_squared]
        ).double()  # (3, parameters)
        per_group = torch.zeros(
            3, len(self.group_names), dtype=torch.float64, device=per_parameter.device
        )  # (3, groups)
        per_group.index_add_(1, self.group_index, per_parameter)

        self._reading = per_group
        self._before = None
        self._armed = False


def module_keys(sums: np.ndarray, group_names: list[str], count: float) -> dict:
    """Norms per group and in total, and the update ratio moved / parameter norm.

    sums is the host copy of the (3 * groups,) accumulated reading, count the number
    of heavy steps behind it. The norm of a group is the root of its squared sum,
    so the total is the norm of the whole parameter vector, not a sum of norms.
    """
    table = sums.reshape(len(MODULE_READINGS), len(group_names)) / count  # (3, groups)
    keys = {}
    for index, group in enumerate(group_names):
        grad_norm = table[0, index] ** 0.5
        param_norm = table[1, index] ** 0.5
        moved_norm = table[2, index] ** 0.5
        keys[f"grad_norm_{group}"] = rounded(grad_norm, 6)
        keys[f"param_norm_{group}"] = rounded(param_norm)
        if param_norm > 0:
            keys[f"update_ratio_{group}"] = rounded(moved_norm / param_norm, 9)

    total_grad, total_param, total_moved = table.sum(axis=1) ** 0.5
    keys["grad_norm_total"] = rounded(total_grad, 6)
    keys["param_norm_total"] = rounded(total_param)
    if total_param > 0:
        keys["update_ratio_total"] = rounded(total_moved / total_param, 9)
    return keys
