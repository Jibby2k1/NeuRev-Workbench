import sys

from neurobench.cli.main import build_parser


def test_pairwise_parser_is_lazy_and_requires_artifact_destination():
    before=set(sys.modules)
    args=build_parser(active_command="experiment").parse_args(["experiment","pairwise-separation","preflight","--config","x.json","--artifact-dir","preflight"])
    assert args.experiment_workflow=="pairwise-separation" and args.experiment_action=="preflight"
    assert "neurobench.experiments.pairwise_separation.runner" not in set(sys.modules)-before
