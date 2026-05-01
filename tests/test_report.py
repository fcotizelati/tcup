import arviz as az
import numpy as np

import tcup
from tcup.cli import main


def _idata():
    rng = np.random.default_rng(24601)
    return az.from_dict(
        posterior={
            "alpha": rng.normal(size=(2, 20)),
            "beta": rng.normal(size=(2, 20, 1)),
            "sigma": rng.lognormal(size=(2, 20)),
            "sigma_68": rng.lognormal(size=(2, 20)),
            "nu": rng.lognormal(size=(2, 20)),
            "outlier_frac": rng.uniform(0, 0.2, size=(2, 20)),
        },
        posterior_predictive={
            "y_scaled": rng.normal(size=(2, 20, 6)),
        },
        observed_data={
            "y_scaled": rng.normal(size=6),
        },
        sample_stats={
            "energy": rng.normal(size=(2, 20)),
        },
    )


def test_write_report_outputs_standard_artifacts(tmp_path):
    artifacts = tcup.write_report(_idata(), tmp_path, save_plots=False)

    assert artifacts["netcdf"].is_file()
    assert artifacts["summary_csv"].is_file()
    assert artifacts["summary_txt"].is_file()
    assert artifacts["report"].is_file()
    assert "posterior" in artifacts["report"].read_text()


def test_write_report_outputs_diagnostic_plots(tmp_path):
    artifacts = tcup.write_report(_idata(), tmp_path, save_netcdf=False)

    plot_names = {path.name for path in artifacts["plots"]}
    assert "trace.png" in plot_names
    assert "posterior.png" in plot_names
    assert "forest.png" in plot_names


def test_cli_reads_csv_and_forwards_report_options(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "x,y,dx,dy\n0.0,1.0,0.1,0.2\n1.0,3.0,0.1,0.2\n2.0,5.0,0.1,0.2\n",
        encoding="utf-8",
    )
    calls = {}

    def fake_tcup(**kwargs):
        calls.update(kwargs)
        return _idata()

    monkeypatch.setattr("tcup.cli.tcup", fake_tcup)

    rc = main(
        [
            str(csv_path),
            "--output-dir",
            str(tmp_path / "report"),
            "--seed",
            "10",
            "--num-samples",
            "11",
            "--no-plots",
        ]
    )

    assert rc == 0
    assert calls["seed"] == 10
    assert calls["num_samples"] == 11
    assert calls["report_kwargs"]["save_plots"] is False
    assert np.allclose(calls["x"], [0.0, 1.0, 2.0])
