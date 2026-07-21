import json
from pathlib import Path

import numpy as np

from neurobench.dynamics.latent_interpretation import build_latent_interpretation_report, build_latent_objective_plan


def test_latent_interpretation_report_exports_separability_neighbors_and_previews(tmp_path):
    codes = []
    vids = []
    labels = []
    centers = {"left": np.array([2.0, 0.0, 0.0], dtype=np.float32), "right": np.array([0.0, 2.0, 0.0], dtype=np.float32)}
    for label, center in centers.items():
        for video_index in range(2):
            vid = f"{video_index}_{label}"
            for frame in range(4):
                codes.append(center + np.array([frame * 0.01, video_index * 0.02, 0.0], dtype=np.float32))
                vids.append(vid)
                labels.append(label)
    latent = tmp_path / "latent_codes.npz"
    np.savez(
        latent,
        latent_codes=np.stack(codes).astype(np.float32),
        frame_video_ids=np.asarray(vids),
        frame_labels=np.asarray(labels),
    )
    run = tmp_path / "autoencoder_run.json"
    run.write_text(json.dumps({"latent_codes_path": str(latent), "checkpoint_path": str(tmp_path / "dummy.pt"), "source_dataset": "unit"}), encoding="utf-8")

    report = build_latent_interpretation_report(autoencoder_run=run, out_dir=tmp_path / "latent_report", max_frame_points=10, nearest_neighbors=1)
    html = Path(report["html_path"]).read_text(encoding="utf-8")
    md = Path(report["markdown_path"]).read_text(encoding="utf-8")

    assert report["frame_count"] == 16
    assert report["video_count"] == 4
    assert report["latent_dim"] == 3
    assert report["label_counts"] == {"left": 8, "right": 8}
    assert report["label_separability"]["nearest_centroid_leave_one_video_accuracy"] == 1.0
    assert report["nearest_neighbors"][0]["neighbors"]
    assert report["top_label_separating_latent_dims"][0]["eta_squared"] > 0.9
    assert len(report["sampled_frame_records"]) <= 10
    for path in report["artifacts"].values():
        assert Path(path).is_file()
        assert Path(path).read_bytes().startswith(b"\x89PNG")
    assert "Latent State Interpretation Report" in html
    assert "Nearest Video Neighbors" in md

def test_latent_objective_plan_uses_weak_separability_evidence(tmp_path):
    report_path = tmp_path / "latent_interpretation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "frame_count": 120,
                "video_count": 6,
                "latent_dim": 8,
                "label_counts": {"left": 40, "neutral": 40, "right": 40},
                "label_separability": {
                    "labels": ["left", "neutral", "right"],
                    "nearest_centroid_leave_one_video_accuracy": 0.2,
                    "between_within_distance_ratio": 0.7,
                    "mean_within_label_distance": 5.0,
                    "mean_between_label_distance": 3.5,
                },
                "top_label_separating_latent_dims": [
                    {"dimension": 6, "eta_squared": 0.36, "label_means": {"left": 0.4, "neutral": -0.7, "right": -0.2}}
                ],
                "report_path": str(report_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plan = build_latent_objective_plan(interpretation_report=report_path, out_dir=tmp_path / "plan")
    saved = json.loads(Path(plan["plan_path"]).read_text(encoding="utf-8"))
    markdown = Path(plan["markdown_path"]).read_text(encoding="utf-8")

    assert saved["diagnosis"]["status"] == "weak_label_separability"
    assert saved["evidence"]["chance_accuracy"] == 1 / 3
    assert saved["recommended_objectives"][0]["name"] == "Held-out-video supervised latent head"
    assert any("held-out videos" in gate.lower() for gate in saved["acceptance_gates"])
    assert "Supervised contrastive latent regularizer" in markdown
    assert "Do not claim left/right/neutral" in markdown
