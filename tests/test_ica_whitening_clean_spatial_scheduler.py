from scripts.rerun_clean_spatial_after_other_geometries import PREREQUISITE_GEOMETRIES


def test_clean_spatial_rerun_waits_for_every_other_geometry() -> None:
    assert PREREQUISITE_GEOMETRIES == (
        "temporal", "joint_spatiotemporal",
        "spatial_then_temporal", "temporal_then_spatial",
    )
