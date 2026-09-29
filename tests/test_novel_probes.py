"""The probe operators of experiments/novel_probes on a 12-block ViT at tiny width.

Each test holds 1 property the README's hypotheses rest on: the attention wrapper
reproduces the module, each mask lands where it claims and nowhere else, every
probe leaves the model exactly as it found it and a seed fixes every draw.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms_v2
from torchvision.models.vision_transformer import VisionTransformer

from experiments.novel_probes import operators
from experiments.novel_probes.measure import choose_rates, hook_count
from models.positions import unplug_dropout

pytestmark = pytest.mark.fast

IMAGE, PATCH, BLOCKS, HEADS, WIDTH = 64, 16, 12, 2, 16
TOKENS = 1 + (IMAGE // PATCH) ** 2


@pytest.fixture(scope="module")
def tiny_vit():
    torch.manual_seed(0)
    network = VisionTransformer(
        image_size=IMAGE,
        patch_size=PATCH,
        num_layers=BLOCKS,
        num_heads=HEADS,
        hidden_dim=WIDTH,
        mlp_dim=2 * WIDTH,
        num_classes=5,
    )
    # torchvision zero-initializes the head, which would make every logit 0.
    nn.init.normal_(network.heads.head.weight, std=1.0)
    model = nn.Sequential(transforms_v2.Resize((IMAGE, IMAGE)), network).eval()
    return model


@pytest.fixture(scope="module")
def images():
    generator = torch.Generator().manual_seed(1)
    batch = torch.randn(4, 3, IMAGE, IMAGE, generator=generator)  # (4, 3, H, W)
    return batch


def stream(batch=4):
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(batch, TOKENS, WIDTH, generator=generator)  # (batch, tokens, width)
    return x


def run(model, images, attach=None, rate=None, seed=0):
    handles = attach(model, rate) if attach else []
    torch.manual_seed(seed)
    try:
        with torch.inference_mode():
            logits = model(images)  # (batch, classes)
    finally:
        unplug_dropout(handles)
    return logits


ATTACHES = {
    "psbd_tm": (operators.attach_psbd_tm, 0.5),
    "middle_band": (operators.attach_middle_band, 0.5),
    "cls_read_knockout": (operators.attach_cls_read_knockout, 0.9),
    "key_mask": (operators.attach_key_mask, 0.5),
    "stratified_token_mask": (operators.attach_stratified_token_mask, 0.5),
    "active_neuron_dropout": (operators.attach_active_neuron_dropout, 0.5),
    "native_jitter": (operators.attach_native_jitter, 1.0),
}


def test_attention_wrapper_with_nothing_masked_reproduces_the_module(tiny_vit):
    attention = operators.vit_blocks(tiny_vit, (1, 1))[0].self_attention
    x = stream()
    with torch.inference_mode():
        reference, _ = attention(x, x, x, need_weights=False)

        def allow_all(batch, heads, tokens, device):
            return torch.ones(batch, heads, tokens, tokens, dtype=torch.bool)

        wrapped, _ = operators.attention_with_mask(attention, allow_all, x, x, x)
    assert wrapped.shape == reference.shape == x.shape
    assert torch.allclose(wrapped, reference, atol=1e-5)


def test_knockout_changes_only_the_class_token_row(tiny_vit):
    attention = operators.vit_blocks(tiny_vit, (9, 9))[0].self_attention
    x = stream()
    torch.manual_seed(0)
    with torch.inference_mode():
        reference, _ = attention(x, x, x, need_weights=False)
        knocked, _ = operators.attention_with_mask(
            attention, lambda *a: operators.cls_read_mask(1.0, *a), x, x, x
        )

        # With every patch edge gone the class token attends to itself only,
        # so its output is its own value vector through out_proj.
        values = F.linear(x, attention.in_proj_weight, attention.in_proj_bias).chunk(
            3, dim=-1
        )[2]  # (batch, tokens, width)
        self_only = attention.out_proj(values[:, 0])  # (batch, width)

    assert torch.allclose(knocked[:, 1:], reference[:, 1:], atol=1e-5)
    assert not torch.allclose(knocked[:, 0], reference[:, 0], atol=1e-3)
    assert torch.allclose(knocked[:, 0], self_only, atol=1e-5)


def test_knockout_attaches_to_the_late_blocks_only(tiny_vit):
    handles = operators.attach_cls_read_knockout(tiny_vit, 0.5)
    wrapped = [
        index + 1
        for index, block in enumerate(operators.vit_blocks(tiny_vit, (1, BLOCKS)))
        if "forward" in block.self_attention.__dict__
    ]
    unplug_dropout(handles)
    assert wrapped == [9, 10, 11, 12]


def test_cls_read_mask_hits_the_requested_share_of_patch_edges():
    torch.manual_seed(0)
    allowed = operators.cls_read_mask(0.7, 64, HEADS, TOKENS, "cpu")
    assert allowed.shape == (64, HEADS, TOKENS, TOKENS)
    assert bool(allowed[:, :, 0, 0].all())
    assert bool(allowed[:, :, 1:, :].all())
    removed = 1.0 - allowed[:, :, 0, 1:].float().mean().item()
    assert abs(removed - 0.7) < 0.03


def test_key_mask_hides_keys_from_every_query_and_keeps_the_class_key():
    torch.manual_seed(0)
    keep = operators.shared_key_mask(0.5, 256, HEADS, TOKENS, "cpu")
    assert keep.shape == (256, 1, 1, TOKENS)
    assert bool(keep[..., 0].all())
    assert abs(1.0 - keep[..., 1:].float().mean().item() - 0.5) < 0.03


def test_key_mask_leaves_masked_tokens_out_of_every_softmax(tiny_vit):
    attention = operators.vit_blocks(tiny_vit, (1, 1))[0].self_attention
    x = stream(batch=2)
    hidden = torch.zeros(2, 1, 1, TOKENS, dtype=torch.bool)
    hidden[..., 0] = True
    hidden[..., 1:3] = True

    # Changing the content of a hidden key token changes nothing for the other
    # queries, which is what "masked as a key only" means.
    changed = x.clone()
    changed[:, 5] = 100.0
    with torch.inference_mode():
        before, _ = operators.attention_with_mask(attention, lambda *a: hidden, x, x, x)
        after, _ = operators.attention_with_mask(
            attention, lambda *a: hidden, changed, changed, changed
        )
    others = [t for t in range(TOKENS) if t != 5]
    assert torch.allclose(before[:, others], after[:, others], atol=1e-5)


def test_stratified_schedule_keeps_every_token_visible_in_every_window():
    torch.manual_seed(0)
    for rate in (0.3, 0.5, 0.6, 0.75):
        schedule = operators.stratified_schedule(512, TOKENS, rate, BLOCKS, "cpu")
        assert schedule.shape == (512, TOKENS, BLOCKS)
        assert not bool(schedule[:, 0].any())

        windows = schedule[:, 1:].reshape(512, TOKENS - 1, 3, operators.STRATIFY_WINDOW)
        visible_per_window = (~windows).sum(dim=-1)  # (batch, patches, windows)
        assert int(visible_per_window.min()) >= 1
        assert abs(schedule[:, 1:].float().mean().item() - rate) < 0.01


def test_stratified_mask_zeroes_the_scheduled_tokens_at_ln_1(tiny_vit, images):
    seen = []
    blocks = operators.vit_blocks(tiny_vit, (1, BLOCKS))
    handles = operators.attach_stratified_token_mask(tiny_vit, 0.5)
    recorders = [
        block.ln_1.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].abs().sum(dim=-1) == 0)
        )
        for block in blocks
    ]
    try:
        torch.manual_seed(0)
        with torch.inference_mode():
            tiny_vit(images)
    finally:
        for handle in recorders:
            handle.remove()
        unplug_dropout(handles)

    zeroed = torch.stack(seen, dim=-1)  # (batch, tokens, blocks)
    assert not bool(zeroed[:, 0].any())
    assert abs(zeroed[:, 1:].float().mean().item() - 0.5) < 0.1
    windows = zeroed[:, 1:].reshape(len(images), TOKENS - 1, 3, 4)
    assert int((~windows).sum(dim=-1).min()) >= 1


def test_active_neuron_mask_never_touches_negative_activations():
    torch.manual_seed(0)
    x = torch.randn(4, TOKENS, 32)  # (batch, tokens, hidden)
    operator = operators.ActiveNeuronMask(0.5).train()
    out = operator(x)
    assert out.shape == x.shape
    negative = x <= 0
    assert torch.equal(out[negative], x[negative])
    positive = x > 0
    assert torch.all(
        (out[positive] == 0) | torch.isclose(out[positive], 2 * x[positive])
    )


def test_native_jitter_keeps_a_constant_image_and_stays_within_the_amplitude():
    x = torch.full((2, 3, 32, 32), 0.7)
    torch.manual_seed(0)
    (jittered,) = operators.native_jitter_hook(1.0, None, (x,))
    assert jittered.shape == x.shape
    assert torch.allclose(jittered, x, atol=1e-6)

    # A horizontal ramp moves by at most the offset, which is 1 native pixel's
    # worth of the ramp at amplitude 1.
    ramp = torch.linspace(0, 31, 32).view(1, 1, 1, 32).expand(1, 3, 32, 32).clone()
    (moved,) = operators.native_jitter_hook(1.0, None, (ramp,))
    shift = (moved - ramp)[..., 1:-1].abs().max().item()
    assert 0.0 < shift <= 0.5 + 1e-4


def test_rollout_ranks_patch_tokens_and_the_top_drop_spares_the_class_token(
    tiny_vit, images
):
    scores = operators.attention_rollout(tiny_vit, images)
    assert scores.shape == (len(images), TOKENS - 1)
    assert bool((scores >= 0).all())

    drop = operators.top_tokens_drop(scores, 3)
    assert drop.shape == (len(images), TOKENS)
    assert not bool(drop[:, 0].any())
    assert drop.sum(dim=1).tolist() == [3] * len(images)
    top = scores.argmax(dim=1)
    assert bool(drop[torch.arange(len(images)), top + 1].all())


def test_fixed_drop_zeroes_exactly_the_chosen_tokens(tiny_vit, images):
    state = {"drop": torch.zeros(len(images), TOKENS, dtype=torch.bool)}
    state["drop"][:, 3] = True
    seen = []
    handles = operators.attach_fixed_token_drop(tiny_vit, state)
    block = operators.vit_blocks(tiny_vit, (7, 7))[0]
    recorder = block.ln_1.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].abs().sum(dim=-1) == 0)
    )
    try:
        with torch.inference_mode():
            tiny_vit(images)
    finally:
        recorder.remove()
        unplug_dropout(handles)
    assert seen[0].nonzero()[:, 1].unique().tolist() == [3]


@pytest.mark.parametrize("name", list(ATTACHES))
def test_every_probe_leaves_the_model_as_it_found_it(tiny_vit, images, name):
    attach, rate = ATTACHES[name]
    hooks_before = hook_count(tiny_vit)
    clean_before = run(tiny_vit, images)

    perturbed = run(tiny_vit, images, attach, rate)

    assert perturbed.shape == clean_before.shape
    assert not torch.allclose(perturbed, clean_before, atol=1e-5), "probe did nothing"
    assert hook_count(tiny_vit) == hooks_before
    assert torch.equal(run(tiny_vit, images), clean_before)


@pytest.mark.parametrize("name", list(ATTACHES))
def test_a_seed_fixes_every_draw(tiny_vit, images, name):
    attach, rate = ATTACHES[name]
    first = run(tiny_vit, images, attach, rate, seed=3)
    second = run(tiny_vit, images, attach, rate, seed=3)
    other = run(tiny_vit, images, attach, rate, seed=4)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_rate_rules_read_the_sign():
    shift = {0.3: 0.02, 0.5: 0.04, 0.7: 0.5, 0.9: 0.85}
    psbd = choose_rates(shift, "psbd")
    assert psbd["primary"] == psbd["adaptive"] == 0.9

    fragile = choose_rates(shift, "fragile")
    assert fragile["stability"] == 0.5
    assert fragile["primary"] == 0.5
    assert fragile["canonical"] == 0.9

    unreached = choose_rates({0.3: 0.1, 0.5: 0.6}, "psbd")
    assert unreached["adaptive"] is None and unreached["primary"] == 0.5
