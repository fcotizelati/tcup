# t-cup - robust linear regression

`t-cup` implements robust Bayesian linear regression for data where both the
independent and dependent variables are measured with uncertainty. The default
backend is the validated NumPyro implementation.

The model uses a Student's t distribution for the intrinsic scatter, Gaussian
measurement errors by default, and an extreme-deconvolution mixture prior for
the latent independent variables. This package bundles
[XDGMM](https://github.com/tholoien/XDGMM), which implements the
[extreme deconvolution algorithm](https://arxiv.org/abs/0905.2979).

## Installation

```bash
pip install tcup
```

The current dependency set supports Python 3.10-3.14. When working from a
source checkout, clone with submodules so the bundled XDGMM package is present:

```bash
git clone --recurse-submodules https://github.com/wm1995/tcup.git
```

## Example

```python
import numpy as np
import tcup

rng = np.random.default_rng(1)
x_true = np.linspace(0, 10, 30)
y_true = 1.0 + 2.0 * x_true + rng.standard_t(df=3, size=x_true.size) * 0.4

dx = np.full_like(x_true, 0.2)
dy = np.full_like(y_true, 0.3)
x_obs = x_true + rng.normal(0, dx)
y_obs = y_true + rng.normal(0, dy)

idata = tcup.tcup(
    x=x_obs,
    y=y_obs,
    dx=dx,
    dy=dy,
    seed=4,
    num_warmup=1000,
    num_samples=1000,
)

print(idata.posterior[["alpha", "beta", "sigma_68"]])
```

Pass `cov_x` instead of `dx` for full covariance matrices with shape
`(N, D, D)`. The number of mixture components in the latent-`x` prior can be
fixed with `x_prior_components`; otherwise XDGMM selects it by BIC.

## Reports and diagnostics

Pass `output_dir` to write a reusable result bundle while still receiving the
ArviZ `InferenceData` object in Python:

```python
idata = tcup.tcup(
    x=x_obs,
    y=y_obs,
    dx=dx,
    dy=dy,
    seed=4,
    output_dir="tcup_report",
)
```

The report directory contains:

- `inference_data.nc`: complete ArviZ result file
- `summary.csv` and `summary.txt`: posterior summaries for key parameters
- `report.md`: a compact Markdown index of groups, variables, and files
- `plots/`: trace, posterior, forest, energy, and posterior-predictive plots

You can also write a report from an existing result:

```python
tcup.write_report(idata, "tcup_report")
```

For CSV data with columns `x`, `y`, `dx`, and `dy`, the command line interface
fits the model and writes the same report bundle:

```bash
tcup-fit data.csv --output-dir tcup_report --seed 4
```

The optional Stan backend can be installed with `pip install "tcup[stan]"`,
but it is experimental and is not guaranteed to match the validated NumPyro
backend.

## Citation

More information can be found in the accompanying paper:

- [RAS Techniques and Instruments](https://doi.org/10.1093/rasti/rzaf035)
- [arXiv](https://arxiv.org/abs/2411.02380)
- [paper repository](https://github.com/wm1995/tcup-paper)

If you find the package useful, please cite:

```bibtex
@article{tcup,
       author = {{Martin}, William and {Mortlock}, Daniel J.},
        title = "{An approach to robust Bayesian regression in astronomy}",
      journal = {RAS Techniques and Instruments},
         year = 2025,
       volume = {4},
          eid = {rzaf035},
          doi = {10.1093/rasti/rzaf035},
archivePrefix = {arXiv},
       eprint = {2411.02380},
 primaryClass = {astro-ph.IM}
}
```
