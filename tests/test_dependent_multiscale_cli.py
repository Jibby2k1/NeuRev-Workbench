from neurobench.cli.main import build_parser


def test_dependent_multiscale_cli_is_lazy_and_has_all_actions():
    parser = build_parser(active_command="experiment")
    synthetic = parser.parse_args([
        "experiment", "dependent-multiscale", "synthetic", "--output-dir", "x"
    ])
    assert synthetic.experiment_action == "synthetic"
    assert synthetic.func.__name__ == "_run_dependent_multiscale_synthetic"
    for action in ("preflight", "run"):
        args = parser.parse_args([
            "experiment", "dependent-multiscale", action, "--config", "x.json"
        ])
        assert args.experiment_action == action
    smoke = parser.parse_args(["experiment", "dependent-multiscale", "smoke"])
    assert smoke.func.__name__ == "_run_dependent_multiscale_smoke"
    report = parser.parse_args([
        "experiment", "dependent-multiscale", "report", "--run-dir", "x"
    ])
    assert report.func.__name__ == "_run_dependent_multiscale_report"
