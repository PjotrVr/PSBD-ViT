"""The token masks experiments/why_token_masking_works/measure.py builds its readings on.

The experiment reasons about which tokens PSBD-TM dropped in each block, so it
runs its own copy of TokenMask that records the draw. That copy is only evidence
about PSBD-TM if it produces the library operator's output bit for bit under the
same seed, alone and plugged into a ViT. The deterministic mask rescales its
survivors the way TokenMask does, and the experiment reads that rescale as inert
at before_attention_norm because LayerNorm removes each token's own scale.
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

from defenses.operators import TokenMask
from experiments.why_token_masking_works.measure import (
    FixedTokenMask,
    PositionDropout,
    RecordingTokenMask,
)
from models.positions import plug_dropout, unplug_dropout

pytestmark = pytest.mark.fast

RATE = 0.54
SEED = 7


def activation(batch=8, tokens=197, channels=16):
    generator = torch.Generator().manual_seed(0)
    # Shifted away from 0 so a zero in the output can only mean a removed token.
    values = torch.rand(batch, tokens, channels, generator=generator) + 0.5
    return values


def tiny_vit():
    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=3,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=5,
    ).eval()
    return model


def test_recording_mask_reproduces_the_library_operator():
    x = activation()  # (8, 197, 16)

    torch.manual_seed(SEED)
    library = TokenMask(RATE).train()(x)  # (8, 197, 16)
    torch.manual_seed(SEED)
    recorder = RecordingTokenMask(RATE).train()
    recorded = recorder(x)  # (8, 197, 16)

    assert torch.equal(library, recorded)
    zeroed = (recorded == 0).all(dim=2)  # (8, 197)
    assert torch.equal(recorder.last_dropped, zeroed)
    assert not recorder.last_dropped[:, 0].any(), "CLS must never be dropped"
    assert 0.3 < recorder.last_dropped[:, 1:].float().mean() < 0.8


def test_recording_mask_reproduces_the_library_operator_inside_a_vit():
    model = tiny_vit()
    images = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(1))

    handles = plug_dropout(
        model,
        "vit",
        ("before_attention_norm",),
        {"before_attention_norm": TokenMask},
        RATE,
    )
    torch.manual_seed(SEED)
    with torch.no_grad():
        library_logits = model(images)  # (4, 5)
    unplug_dropout(handles)

    recorders = []

    def build(rate):
        recorders.append(RecordingTokenMask(rate))
        return recorders[-1]

    handles = plug_dropout(
        model, "vit", ("before_attention_norm",), {"before_attention_norm": build}, RATE
    )
    torch.manual_seed(SEED)
    with torch.no_grad():
        recorded_logits = model(images)  # (4, 5)
    unplug_dropout(handles)

    assert len(recorders) == 3
    assert torch.equal(library_logits, recorded_logits)
    assert all(r.last_dropped.shape == (4, 17) for r in recorders)


def test_fixed_mask_rescale_is_inert_under_layernorm():
    x = activation()  # (8, 197, 16)
    positions = torch.tensor([3, 50, 180])
    norm = nn.LayerNorm(16)

    masked = FixedTokenMask(positions)(x)  # (8, 197, 16)
    unscaled = x.clone()
    unscaled[:, positions, :] = 0.0

    assert (masked[:, positions, :] == 0).all()
    assert torch.equal(masked[:, 0], x[:, 0]), "CLS is left at scale 1"
    assert not torch.allclose(masked[:, 1], x[:, 1]), "survivors are rescaled"
    assert torch.allclose(norm(masked), norm(unscaled), atol=1e-5)


def test_position_dropout_touches_only_its_positions():
    x = activation()  # (8, 197, 16)
    positions = torch.tensor([10, 11])

    torch.manual_seed(SEED)
    out = PositionDropout(0.5, positions)(x)  # (8, 197, 16)

    untouched = torch.ones(197, dtype=torch.bool)
    untouched[positions] = False
    assert torch.equal(out[:, untouched], x[:, untouched])
    assert (out[:, positions] == 0).any()
