import os

import pytest

from neurobench.experiments.information_source_separation.cnmf_adapter import (
    audit_caiman_backend,
)


def test_pinned_external_caiman_environment_is_auditable() -> None:
    interpreter = os.environ.get("NEUROBENCH_CAIMAN_PYTHON")
    if not interpreter:
        pytest.skip(
            "set NEUROBENCH_CAIMAN_PYTHON to exercise the optional pinned CaImAn integration"
        )

    audit = audit_caiman_backend("1.13.1", python_executable=interpreter)
    assert audit["available"]
    assert audit["version_frozen"]
    assert audit["fit_authorized"]
    assert audit["environment_kind"] == "isolated_external_python"
