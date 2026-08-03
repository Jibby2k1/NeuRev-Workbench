import json

import numpy as np

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_config import (
    DependentMultiscaleConfig,
)
from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_program import preflight


def test_preflight_validates_alignment_and_writes_label_projection(tmp_path):
    rng = np.random.default_rng(12)
    observation = rng.normal(size=(12, 20, 21)).astype(np.float32)
    np.save(tmp_path / "observation.npy", observation)
    np.save(tmp_path / "carrier.npy", observation)
    (tmp_path / "labels.tsv").write_text(
        "burst_id\tx_px\ty_px\n1\t10\t9\n", encoding="utf-8"
    )
    raw = json.loads(
        open("examples/spon_ca_burst_dependent_multiscale_v1.example.json", encoding="utf-8").read()
    )
    raw["input"].update({
        "observation_npy": "observation.npy",
        "display_observation_npy": "observation.npy",
        "scientific_carrier_npy": "carrier.npy",
        "labels_tsv": "labels.tsv",
    })
    raw["frames"] = {"review_start_ui": 1, "review_end_ui": 12, "quiet_count": 4}
    raw["preflight_dir"] = "preflight"
    raw["output_dir"] = "run"
    manifest = tmp_path / "config.json"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    config = DependentMultiscaleConfig.load(manifest)
    audit = preflight(config)
    assert audit["ready"]
    assert all(type(value) is bool for value in audit["gates"].values())
    assert audit["label_projection_count"] == 1
    assert (config.preflight_dir / "label_projection_overlay.png").is_file()
    assert not config.output_dir.exists()
