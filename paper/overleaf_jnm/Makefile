.PHONY: story story-check story-live-check figures diagnostics package

PYTHON ?= ../../.venv-neurobench/bin/python

story:
	PYTHONPATH=../.. $(PYTHON) -m neurobench.research.registry build
	$(PYTHON) scripts/build_research_story.py

story-check:
	PYTHONPATH=../.. $(PYTHON) -m neurobench.research.registry check
	$(PYTHON) scripts/build_research_story.py --check

story-live-check:
	PYTHONPATH=../.. $(PYTHON) -m neurobench.research.registry verify-live --check
	$(PYTHON) scripts/build_research_story.py --check --verify-live-artifacts

figures:
	PYTHONPATH=../.. $(PYTHON) scripts/build_fig04_population_trace_summary.py
	PYTHONPATH=../.. $(PYTHON) scripts/build_fig05_pipeline_diagnostic_audit.py
	PYTHONPATH=../.. $(PYTHON) scripts/build_fig06_ica_model_internals.py
	PYTHONPATH=../.. $(PYTHON) scripts/build_fig07_ls_denominator_diagnostics.py
	PYTHONPATH=../.. $(PYTHON) scripts/import_fig08_full_trace_feature_panel.py
	PYTHONPATH=../.. $(PYTHON) scripts/import_fig09_automated_feature_validation.py
	PYTHONPATH=../.. $(PYTHON) scripts/import_fig11_feature_deep_dives.py

diagnostics: figures story

package: story-check
	$(PYTHON) scripts/build_overleaf_upload.py
