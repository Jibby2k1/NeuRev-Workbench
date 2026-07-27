from __future__ import annotations

import json


def test_pure_label_reconciliation_preserves_rows_and_projection_contract(tmp_path):
    from neurobench.workbench.label_reconciliation import reconcile_label_table

    source = tmp_path / "labels.tsv"
    source.write_text(
        "roi_id\tx\ty\tstart_frame\tend_frame\tlabel\tconfidence\n"
        "r1\t1\t1\t1\t2\tneuron\t0.9\n"
        "r1\t2\t2\t2\t3\tduplicate\t0.8\n"
        "r2\t3\t2\t1\t1\tunsure\t0.4\n"
        "\t1\t1\t0\t2\tbad\t0.5\n",
        encoding="utf-8",
    )
    review = {
        "video": {"frames": 3, "width": 5, "height": 4},
        "rois": [{"id": "r1"}],
    }
    record = {
        "dataset_id": "demo",
        "import_id": "imp_labels",
        "source_path": "Inputs/demo/labels.tsv",
        "destination_path": "Inputs/demo/labels.tsv",
        "checksum": {"sha256": "a" * 64, "size_bytes": source.stat().st_size},
        "metadata": {
            "columns": [
                "roi_id",
                "x",
                "y",
                "start_frame",
                "end_frame",
                "label",
                "confidence",
            ],
            "row_count": 4,
        },
    }

    result = reconcile_label_table(
        source=source,
        review=review,
        import_record=record,
    )

    assert result.summary == {
        "total_rows": 4,
        "matched_rows": 2,
        "unmatched_rows": 1,
        "duplicate_rows": 1,
        "rejected_rows": 1,
    }
    assert json.loads(result.artifact_bytes) == result.artifact
    assert list(result.artifact["rows"]) == [
        "row_00000002",
        "row_00000003",
        "row_00000004",
        "row_00000005",
    ]
    assert result.artifact["rows"]["row_00000003"]["reconciliation"]["classifications"] == [
        "matched",
        "duplicate",
    ]
    assert result.artifact["rows"]["row_00000004"]["reconciliation"]["status"] == "unmatched"
    assert result.artifact["rows"]["row_00000005"]["reconciliation"]["status"] == "rejected"
    assert result.artifact["coordinate_contract"]["frames"] == "one-based inclusive UI frames"
    assert result.artifact["projection"] == {"point_count": 4, "truncated": False}
    assert b"external-label projection; points=4; truncated=false" in result.overlay_svg
    assert list(tmp_path.iterdir()) == [source]


def test_pure_label_reconciliation_bounds_projection_without_dropping_rows(tmp_path):
    from neurobench.workbench.label_reconciliation import reconcile_label_table

    source = tmp_path / "labels.tsv"
    source.write_text("roi_id\tx\ty\nr1\t1\t1\nr2\t2\t2\n", encoding="utf-8")
    result = reconcile_label_table(
        source=source,
        review={"video": {"frames": 1, "width": 4, "height": 4}, "rois": []},
        import_record={
            "dataset_id": "demo",
            "import_id": "imp_labels",
            "metadata": {"columns": ["roi_id", "x", "y"], "row_count": 2},
        },
        max_overlay_points=1,
    )

    assert result.summary["total_rows"] == 2
    assert len(result.artifact["rows"]) == 2
    assert result.artifact["projection"] == {"point_count": 1, "truncated": True}
