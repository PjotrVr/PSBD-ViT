"""Novel PSBD-like probes for ViT-B/16, each attached by hooks or a removable wrapper.

Every attach function takes the loaded Sequential(Resize, VisionTransformer) and a
rate, mutates no weight and returns handles that models.positions.unplug_dropout
removes. The README states each probe's hypothesis and what would refute it.
"""

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from defenses.operators import TokenMask
from models.backbones import network_core
from models.positions import ForwardRestore, plug_dropout

# Blocks are 1-indexed and inclusive, as models.positions.plug_dropout reads them.
LATE_BLOCKS = (9, 12)
EARLY_BLOCKS = (1, 4)
# The window of consecutive blocks within which a stratified mask keeps every
# token visible at least once. 4 is the size of the late reading band.
STRATIFY_WINDOW = 4


def attach_psbd_tm(model, rate):
    handles = plug_dropout(
        model,
        "vit",
        ("before_attention_norm",),
        {"before_attention_norm": TokenMask},
        rate,
    )
    return handles


def attach_middle_band(model, rate):
    handles = plug_dropout(
        model,
        "vit",
        ("before_attention_residual", "before_mlp_residual"),
        {},
        rate,
        block_range=(5, 8),
    )
    return handles


def attach_cls_read_knockout(model, rate, blocks=LATE_BLOCKS):
    # Only the class token's query loses keys. Every patch token still attends
    # to every other and keeps its residual entry, so the knockout attacks the
    # read and leaves storage alone.
    handles = [
        wrap_attention(block, functools.partial(cls_read_mask, rate))
        for block in vit_blocks(model, blocks)
    ]
    return handles


def attach_key_mask(model, rate):
    handles = [
        wrap_attention(block, functools.partial(shared_key_mask, rate))
        for block in vit_blocks(model, (1, 12))
    ]
    return handles


def attach_stratified_token_mask(model, rate):
    blocks = vit_blocks(model, (1, 12))
    assert rate <= (STRATIFY_WINDOW - 1) / STRATIFY_WINDOW, rate
    assert len(blocks) % STRATIFY_WINDOW == 0, len(blocks)

    # The 12 hooks share 1 schedule, drawn when block 1 fires, so each forward
    # pass gets a fresh schedule and every block reads its own column of it.
    state = {"schedule": None}
    handles = [
        block.ln_1.register_forward_pre_hook(
            functools.partial(stratified_hook, state, index, rate, len(blocks))
        )
        for index, block in enumerate(blocks)
    ]
    return handles


def attach_active_neuron_dropout(model, rate):
    handles = plug_dropout(
        model, "vit", ("mlp_neurons",), {"mlp_neurons": ActiveNeuronMask}, rate
    )
    return handles


def attach_fixed_token_drop(model, state):
    # state["drop"] is a (batch, tokens) bool set by the caller per batch, so a
    # deterministic mask can depend on the image (rollout ranking) or on a seed.
    handles = [
        block.ln_1.register_forward_pre_hook(functools.partial(fixed_drop_hook, state))
        for block in vit_blocks(model, (1, 12))
    ]
    return handles


def attach_native_jitter(model, amplitude):
    # model is Sequential(Resize, network), so a pre-hook on the root sees the
    # image at its native resolution, where the WaNet trigger and its noise
    # mode were defined.
    handle = model.register_forward_pre_hook(
        functools.partial(native_jitter_hook, amplitude)
    )
    handles = [handle]
    return handles


def vit_blocks(model, blocks):
    layers = list(network_core(model).encoder.layers)
    first, last = blocks
    assert 1 <= first <= last <= len(layers), (blocks, len(layers))
    selected = layers[first - 1 : last]
    return selected


def wrap_attention(block, build_allowed):
    attention = block.self_attention
    attention.forward = functools.partial(attention_with_mask, attention, build_allowed)
    handle = ForwardRestore(attention)
    return handle


def attention_with_mask(attention, build_allowed, query, key, value, **kwargs):
    # EncoderBlock calls self_attention(x, x, x, need_weights=False), plain
    # self-attention with no mask of its own, which is all this reproduces.
    assert query is key and query is value, "self-attention only"
    batch, tokens, channels = query.shape
    heads = attention.num_heads
    head_dim = channels // heads

    projected = F.linear(
        query, attention.in_proj_weight, attention.in_proj_bias
    )  # (batch, tokens, 3 * channels)
    q, k, v = [
        part.view(batch, tokens, heads, head_dim).transpose(1, 2)
        for part in projected.chunk(3, dim=-1)
    ]  # each (batch, heads, tokens, head_dim)

    allowed = build_allowed(
        batch, heads, tokens, query.device
    )  # bool, broadcastable to (batch, heads, tokens, tokens)
    attended = F.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed
    )  # (batch, heads, tokens, head_dim)

    merged = attended.transpose(1, 2).reshape(
        batch, tokens, channels
    )  # (batch, tokens, channels)
    output = attention.out_proj(merged)  # (batch, tokens, channels)
    return output, None


def cls_read_mask(rate, batch, heads, tokens, device):
    # Each head's edge from the class token to each patch key is removed
    # independently, so a trigger survives a block when any head keeps any of
    # its keys. The class token's edge to itself always stays, which keeps the
    # softmax row defined when every patch edge is gone.
    keep_patch = (
        torch.rand(batch, heads, tokens - 1, device=device) >= rate
    )  # (batch, heads, tokens - 1)
    allowed = torch.ones(
        batch, heads, tokens, tokens, dtype=torch.bool, device=device
    )  # (batch, heads, tokens, tokens)
    allowed[:, :, 0, 1:] = keep_patch
    return allowed


def shared_key_mask(rate, batch, heads, tokens, device):
    # A masked token disappears as a key for every query and every head, and
    # keeps its own query and residual entry. Unlike a zeroed token it leaves
    # no shared key behind, so no sink forms (E6).
    keep = (
        torch.rand(batch, 1, 1, tokens, device=device) >= rate
    )  # (batch, 1, 1, tokens)
    keep[..., 0] = True
    return keep


def stratified_schedule(batch, tokens, rate, num_blocks, device):
    # Within each window of STRATIFY_WINDOW blocks a token is masked in exactly
    # floor(rate * window) blocks, plus 1 more with the leftover probability,
    # so its marginal rate is `rate` and it is visible in at least 1 block of
    # every window whenever rate <= (window - 1) / window.
    windows = num_blocks // STRATIFY_WINDOW
    base_count = math.floor(rate * STRATIFY_WINDOW)
    leftover = rate * STRATIFY_WINDOW - base_count
    counts = (
        base_count
        + (torch.rand(batch, tokens, windows, device=device) < leftover).long()
    )  # (batch, tokens, windows)

    # The rank of uniform noise is a uniform random order of the window's blocks.
    block_rank = (
        torch.rand(batch, tokens, windows, STRATIFY_WINDOW, device=device)
        .argsort(dim=-1)
        .argsort(dim=-1)
    )  # (batch, tokens, windows, window)
    masked = block_rank < counts.unsqueeze(-1)  # (batch, tokens, windows, window)

    schedule = masked.reshape(batch, tokens, num_blocks)  # (batch, tokens, blocks)
    schedule[:, 0, :] = False
    return schedule


def stratified_hook(state, index, rate, num_blocks, module, args):
    x = args[0]  # (batch, tokens, channels)
    batch, tokens, _ = x.shape
    if index == 0:
        state["schedule"] = stratified_schedule(
            batch, tokens, rate, num_blocks, x.device
        )
    schedule = state["schedule"]
    assert schedule.shape[:2] == (batch, tokens), (schedule.shape, x.shape)

    # The same inverted scale as TokenMask, which ln_1 removes anyway since a
    # LayerNorm is invariant to the scale of its input.
    keep = (~schedule[:, :, index]).unsqueeze(-1).to(x.dtype)  # (batch, tokens, 1)
    masked = x * keep / (1.0 - rate)  # (batch, tokens, channels)
    return (masked,) + tuple(args[1:])


def fixed_drop_hook(state, module, args):
    x = args[0]  # (batch, tokens, channels)
    drop = state["drop"]  # (batch, tokens)
    assert drop.shape == x.shape[:2], (drop.shape, x.shape)
    assert not bool(drop[:, 0].any()), "the class token is never dropped"

    masked = x * (~drop).unsqueeze(-1).to(x.dtype)  # (batch, tokens, channels)
    return (masked,) + tuple(args[1:])


class ActiveNeuronMask(nn.Module):
    # The plain channel mask at mlp_neurons restricted to positive activations:
    # a hidden unit's mask is shared across the sample's tokens as in
    # GroupChannelMask, and it only removes the unit where GELU let it fire.
    def __init__(self, rate):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            return x
        batch, _, channels = x.shape  # (batch, tokens, hidden)
        keep = (torch.rand(batch, 1, channels, device=x.device) >= self.rate).to(
            x.dtype
        )  # (batch, 1, hidden)

        scaled_positive = x * keep / (1.0 - self.rate)  # (batch, tokens, hidden)
        out = torch.where(x > 0, scaled_positive, x)  # (batch, tokens, hidden)
        return out


def native_jitter_hook(amplitude, module, args):
    x = args[0]  # (batch, channels, height, width), normalized, native resolution
    batch, _, height, width = x.shape
    assert height == width, x.shape

    # Bilinear resampling is linear and border padding copies edge values, so
    # jittering the normalized image equals jittering the pixels and
    # normalizing afterwards. The offset scale 1 / h in [-1, 1] coordinates is
    # the WaNet noise mode's (attacks/wanet.py, apply_cover).
    identity = torch.eye(2, 3, device=x.device, dtype=x.dtype).expand(
        batch, 2, 3
    )  # (batch, 2, 3)
    grid = F.affine_grid(
        identity, list(x.shape), align_corners=True
    )  # (batch, height, width, 2)
    offsets = (
        (torch.rand(batch, height, width, 2, device=x.device, dtype=x.dtype) * 2 - 1)
        * amplitude
        / height
    )  # (batch, height, width, 2)

    jittered = F.grid_sample(
        x, grid + offsets, mode="bilinear", padding_mode="border", align_corners=True
    )  # (batch, channels, height, width)
    return (jittered,) + tuple(args[1:])


def attention_rollout(model, images):
    # Abnar and Zuidema's rollout: each block's head-averaged attention mixed
    # with the identity for the residual path, row-normalized and multiplied
    # through depth. Row 0 of the product says how much each patch token
    # contributes to the class token at the output.
    blocks = vit_blocks(model, (1, 12))
    captured = []
    handles = [
        wrap_attention_capture(block.self_attention, captured) for block in blocks
    ]
    try:
        with torch.inference_mode():
            model(images)
    finally:
        for handle in handles:
            handle.remove()

    batch, tokens, _ = captured[0].shape
    identity = torch.eye(tokens, device=captured[0].device).expand(
        batch, tokens, tokens
    )  # (batch, tokens, tokens)
    rollout = identity
    for attention in captured:
        mixed = 0.5 * attention.float() + 0.5 * identity  # (batch, tokens, tokens)
        mixed = mixed / mixed.sum(dim=-1, keepdim=True)
        rollout = mixed @ rollout  # (batch, tokens, tokens)

    cls_to_patch = rollout[:, 0, 1:]  # (batch, tokens - 1)
    return cls_to_patch


def wrap_attention_capture(attention, captured):
    attention.forward = functools.partial(
        attention_capturing_weights, attention, captured
    )
    handle = ForwardRestore(attention)
    return handle


def attention_capturing_weights(attention, captured, query, key, value, **kwargs):
    assert query is key and query is value, "self-attention only"
    batch, tokens, channels = query.shape
    heads = attention.num_heads
    head_dim = channels // heads

    projected = F.linear(
        query, attention.in_proj_weight, attention.in_proj_bias
    )  # (batch, tokens, 3 * channels)
    q, k, v = [
        part.view(batch, tokens, heads, head_dim).transpose(1, 2)
        for part in projected.chunk(3, dim=-1)
    ]  # each (batch, heads, tokens, head_dim)

    weights = torch.softmax(
        (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim), dim=-1
    )  # (batch, heads, tokens, tokens)
    captured.append(weights.mean(dim=1))  # (batch, tokens, tokens)

    attended = weights.to(v.dtype) @ v  # (batch, heads, tokens, head_dim)
    merged = attended.transpose(1, 2).reshape(
        batch, tokens, channels
    )  # (batch, tokens, channels)
    output = attention.out_proj(merged)  # (batch, tokens, channels)
    return output, None


def top_tokens_drop(scores, count):
    # scores is (batch, tokens - 1) over patch tokens. The class token column is
    # prepended as never dropped.
    batch, patches = scores.shape
    top = scores.topk(count, dim=1).indices  # (batch, count)
    drop_patch = torch.zeros(batch, patches, dtype=torch.bool, device=scores.device)
    drop_patch.scatter_(1, top, True)  # (batch, tokens - 1)

    keep_cls = torch.zeros(batch, 1, dtype=torch.bool, device=scores.device)
    drop = torch.cat([keep_cls, drop_patch], dim=1)  # (batch, tokens)
    return drop


def random_tokens_drop(batch, tokens, count, device):
    # The top `count` of uniform noise is a uniform random subset of that size,
    # drawn from the global generator the caller seeds.
    noise = torch.rand(batch, tokens - 1, device=device)  # (batch, tokens - 1)
    drop = top_tokens_drop(noise, count)  # (batch, tokens)
    return drop
