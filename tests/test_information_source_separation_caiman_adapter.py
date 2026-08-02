from neurobench.experiments.information_source_separation.cnmf_adapter import (
    audit_caiman_backend,
)


def test_pinned_external_caiman_environment_is_auditable() -> None:
    audit = audit_caiman_backend("1.13.1")
    assert audit["available"]
    assert audit["version_frozen"]
    assert audit["fit_authorized"]
    assert audit["environment_kind"] == "isolated_external_python"
