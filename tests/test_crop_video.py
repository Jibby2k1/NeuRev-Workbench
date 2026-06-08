import json

import numpy as np
import tifffile

from neurobench.data.crop import CropBox, crop_video_stack, write_crop_manifest
from neurobench.data.video import video_metadata


def test_crop_video_stack_writes_exact_half_open_crop(tmp_path):
    source = tmp_path / "source.tif"
    out = tmp_path / "cropped.tif"
    arr = np.arange(4 * 8 * 9, dtype=np.uint16).reshape(4, 8, 9)
    tifffile.imwrite(source, arr)

    summary = crop_video_stack(source_path=source, output_path=out, crop=CropBox(2, 1, 7, 6), chunk_size_frames=2)

    cropped = tifffile.imread(out)
    assert cropped.shape == (4, 5, 5)
    np.testing.assert_array_equal(cropped, arr[:, 1:6, 2:7])
    assert summary["source_shape"] == [4, 8, 9]
    assert summary["output_shape"] == [4, 5, 5]
    assert summary["crop_box"] == {"x0": 2, "y0": 1, "x1": 7, "y1": 6, "width": 5, "height": 5}
    assert video_metadata(out)["dtype"] == "uint16"


def test_write_crop_manifest_records_videos_and_crop(tmp_path):
    manifest_path = tmp_path / "crop_manifest.json"
    crop = CropBox(81, 115, 593, 627)

    payload = write_crop_manifest(summaries=[{"source_path": "a.tif", "output_path": "b.tif"}], output_path=manifest_path, crop=crop)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == loaded
    assert loaded["video_count"] == 1
    assert loaded["crop_box"]["width"] == 512
    assert loaded["crop_box"]["height"] == 512
