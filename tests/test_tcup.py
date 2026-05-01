import arviz as az
from jax import config

import tcup

config.update("jax_enable_x64", True)


def test_tcup(data):
    mcmc = tcup.tcup(
        **data,
        num_warmup=25,
        num_samples=25,
        num_chains=1,
        prior_samples=25,
    )
    assert isinstance(mcmc, az.InferenceData)


def test_tcup_writes_report(data, tmp_path):
    mcmc = tcup.tcup(
        **data,
        num_warmup=25,
        num_samples=25,
        num_chains=1,
        prior_samples=25,
        output_dir=tmp_path,
        report_kwargs={"save_plots": False},
    )

    assert isinstance(mcmc, az.InferenceData)
    assert (tmp_path / "inference_data.nc").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "report.md").is_file()


def test_ncup(data):
    mcmc = tcup.tcup(
        **data,
        model="ncup",
        num_warmup=25,
        num_samples=25,
        num_chains=1,
        prior_samples=25,
    )
    assert isinstance(mcmc, az.InferenceData)


def test_fixed(data):
    mcmc = tcup.tcup(
        **data,
        model="fixed",
        shape_param=3,
        num_warmup=25,
        num_samples=25,
        num_chains=1,
        prior_samples=25,
    )
    assert isinstance(mcmc, az.InferenceData)
