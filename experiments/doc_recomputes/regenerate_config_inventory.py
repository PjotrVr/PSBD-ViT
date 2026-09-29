# Regenerates docs/psbd-vit-configs-tested.md with scripts/vit_config_inventory.py.
# The declaration renamed its basis keys position_config and perturbation to position
# and operator after the generator was written, so the old names are added back in
# memory here and the generator itself is left untouched.
import sys

sys.path.insert(0, "scripts")
import vit_config_inventory as inventory  # noqa: E402

declaration = inventory.load("configs/psbd_basis.json")
for entry in declaration["basis"]:
    entry.setdefault("position_config", entry.get("position"))
    entry.setdefault("perturbation", entry.get("operator"))
text = inventory.build(inventory.load("results/coverage/coverage.json"), declaration)
with open(sys.argv[1], "w") as handle:
    handle.write(text)
