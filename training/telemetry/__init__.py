"""Training telemetry: how a backdoor builds up over training, 1 JSONL line per window.

Ported in pattern from the windowed telemetry of the ek_solver project
(solver/dmc/telemetry and solver/dmc/dormancy.py). Every per-step statistic is a
mean over a window of steps. The step hooks add into tensors that stay on the
device, nothing is read back while a window is open, and a flush is 1 host
synchronization that concatenates every accumulator and copies it once. A hook
that synchronized per step would cost more than everything it measures.

3 cadences, because the probes do not cost the same.

- The light window (every --telemetry-every steps, 25 by default) carries the
  clean, poisoned and cover loss, train accuracy and logit margin, computed from
  the logits the update already holds.
- The heavy step (every --telemetry-heavy-every steps, 100 by default, always
  the last step of a light window) adds per-module gradient and parameter norms
  with the measured update ratio, and dormant-unit shares and effective ranks of
  the MLP hidden layers and the classifier features on a fixed held-out batch.
- The epoch record reads ASR and clean accuracy on fixed held-out pairs, margin
  retention under PSBD-TM at fixed rates and the backdoor-direction surplus
  factor at the last block.

A probe that is gated off writes no key, never a 0 that reads as a measurement.

Layout: window (the accumulators, the single transfer and the file), rows (the
light loss split), modules (norms and update ratio), dormancy, heldout (the
pairs and the epoch readings), surplus (the surplus factor) and record (the
Telemetry the loop calls). Import the submodule you need. This package
re-exports nothing.
"""
