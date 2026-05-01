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
