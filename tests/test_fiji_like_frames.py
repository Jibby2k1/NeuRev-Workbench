from __future__ import annotations

import json

import numpy as np

from tools.build_fiji_like_frames import render_fiji_like_frames


def test_render_fiji_like_frames_writes_stack_window_summary(tmp_path):
    stack = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    source = tmp_path / "stack.npy"
    out_dir = tmp_path / "frames"
    np.save(source, stack)

    summary = render_fiji_like_frames(
        source_npy=source,
        out_dir=out_dir,
        low_percentile=0.0,
        high_percentile=100.0,
        sample_frames=3,
    )

    assert summary["rendering"] == "fiji_like_stack_window"
    assert summary["frame_count"] == 3
    assert summary["width"] == 5
    assert summary["height"] == 4
    assert summary["low"] == 0.0
    assert summary["high"] == 59.0
    assert (out_dir / "frame_001.png").exists()
    assert (out_dir / "frame_003.png").exists()

    written = json.loads((out_dir / "display_window.json").read_text(encoding="utf-8"))
    assert written["frame_pattern"] == "frame_%03d.png"
    assert written["sample_frame_count"] == 3
