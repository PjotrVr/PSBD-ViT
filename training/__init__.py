"""Training a backbone on a poisoned dataset, and the checkpoint it leaves behind.

loop runs the epochs and writes the checkpoint with its provenance sidecar. sam is
the Sharpness-Aware Minimization wrapper, used on top of AdamW when a run asks for
it. telemetry is the optional per-window and per-epoch record of how the backdoor
builds up during training, off unless a run asks for it.

Import the submodule you need. This package re-exports nothing.
"""
