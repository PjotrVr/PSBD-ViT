"""The confidence null against Hendrycks and Gimpel's maximum softmax probability code.

The null has no backdoor paper. Its statistic is the maximum softmax
probability baseline of Hendrycks and Gimpel (ICLR 2017), whose released code is
pinned as third_party/error-detection. The vision script reads the confidence
as T.max(test_prediction, axis=1) over a Theano softmax
(Vision/CIFAR_Detection.py line 252), which cannot run here. The ASR script
of the same repository computes the same softmax in numpy
(ASR/CTC/CTC_eval.py lines 115 to 117). That numpy softmax is executed here on
the tiny model's logits, its row maximum taken as the vision script takes it,
and the port must return exactly its negation, to 1e-6 in float32.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from detectors.confidence import confidence_scores
from tests.reference.third_party import load_definitions, reference_file

DEVICE = torch.device("cpu")
NUM_CLASSES = 7


def load_reference_softmax():
    path = reference_file("error-detection", "ASR", "CTC", "CTC_eval.py")
    namespace = load_definitions(path, ("softmax",), {"np": np})
    return namespace["softmax"]


def tiny_classifier() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, NUM_CLASSES)).eval()
    # Spread the logits so some rows are confident and some are not.
    with torch.no_grad():
        model[1].weight.mul_(4.0)
    return model


def test_matches_the_maximum_softmax_probability_negated_to_1e_6():
    reference_softmax = load_reference_softmax()
    model = tiny_classifier()
    generator = torch.Generator().manual_seed(1)
    images = torch.randn(10, 3, 4, 4, generator=generator)
    loader = DataLoader(TensorDataset(images, torch.zeros(10)), batch_size=4)

    ours = confidence_scores(model, loader, DEVICE, use_bfloat16=False)  # (10,)

    with torch.no_grad():
        logits = model(images).double().numpy()  # (10, NUM_CLASSES)
    maximum_softmax = np.max(reference_softmax(logits), axis=1)  # (10,)

    assert maximum_softmax.min() < 0.6 < maximum_softmax.max()
    # The single deliberate difference is the sign, applied once so that low
    # means poisoned as the registry requires.
    assert np.allclose(ours.numpy(), -maximum_softmax, atol=1e-6)
