"""Reporting helpers for t-cup fit results."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

DEFAULT_REPORT_VARS = (
    "alpha",
    "beta",
    "sigma",
    "sigma_68",
    "nu",
    "outlier_frac",
)


def _prefixed(file_prefix: str, name: str) -> str:
    return f"{file_prefix}{name}"


def _selected_vars(
    idata: Any,
    var_names: Optional[Sequence[str]],
) -> Optional[list[str]]:
    if var_names is None or not hasattr(idata, "posterior"):
        return None

    posterior = idata.posterior
    selected = [name for name in var_names if name in posterior.data_vars]
    return selected or None


def _dims_report(idata: Any) -> list[str]:
    if not hasattr(idata, "posterior"):
        return []

    lines = []
    for name, data_array in idata.posterior.data_vars.items():
        shape = " x ".join(str(size) for size in data_array.shape)
        dims = ", ".join(data_array.dims)
        lines.append(f"- `{name}`: {shape} ({dims})")
    return lines


def _write_plot(path: Path, plot_func, warnings: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    try:
        plt.close("all")
        plot_func()
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception as exc:  # pragma: no cover - backend dependent
        warnings.append(f"{path.name}: {exc}")
    finally:
        plt.close("all")


def _as_optional_1d(value: Any) -> Any:
    if value is None:
        return None

    import numpy as np

    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    return array


def _regression_plot(
    idata: Any,
    *,
    x: Any,
    y: Any,
    dy: Any = None,
    cov_x: Any = None,
    max_draws: int = 100,
    random_seed: int = 12345,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = _as_optional_1d(x)
    y = _as_optional_1d(y)
    dy = _as_optional_1d(dy)
    if x is None or y is None:
        raise ValueError("x and y are required for a regression plot")
    if x.ndim != 1:
        raise ValueError("regression plot is only supported for 1D x")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y must be a 1D array with the same length as x")
    if "alpha" not in idata.posterior or "beta" not in idata.posterior:
        raise ValueError("posterior must contain alpha and beta")

    beta = np.asarray(idata.posterior["beta"].values)
    beta = beta.reshape(-1, beta.shape[-1])
    if beta.shape[1] != 1:
        raise ValueError("regression plot is only supported for 1D beta")
    beta = beta[:, 0]

    alpha = np.asarray(idata.posterior["alpha"].values).reshape(-1)
    grid = np.linspace(x.min(), x.max(), 200)
    y_draws = alpha[:, np.newaxis] + beta[:, np.newaxis] * grid[np.newaxis, :]

    rng = np.random.default_rng(random_seed)
    n_draws = min(max_draws, y_draws.shape[0])
    draw_idx = rng.choice(y_draws.shape[0], size=n_draws, replace=False)

    q025, q16, q50, q84, q975 = np.quantile(
        y_draws,
        [0.025, 0.16, 0.5, 0.84, 0.975],
        axis=0,
    )

    xerr = None
    if cov_x is not None:
        cov_x = np.asarray(cov_x, dtype=float)
        if cov_x.shape == (x.shape[0], 1, 1):
            xerr = np.sqrt(cov_x[:, 0, 0])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(
        grid,
        q025,
        q975,
        color="tab:red",
        alpha=0.12,
        label="95% regression interval",
    )
    ax.fill_between(
        grid,
        q16,
        q84,
        color="tab:red",
        alpha=0.25,
        label="68% regression interval",
    )
    ax.plot(grid, q50, color="tab:red", lw=2, label="posterior median")
    for idx in draw_idx:
        ax.plot(grid, y_draws[idx], color="tab:red", alpha=0.08, lw=0.8)
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=dy,
        fmt="o",
        ms=4,
        color="black",
        ecolor="0.45",
        elinewidth=0.8,
        capsize=0,
        label="observed data",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize="small")
    ax.set_title("Posterior regression")
    fig.tight_layout()


def _write_plots(
    idata: Any,
    output_dir: Path,
    var_names: Optional[Sequence[str]],
    file_prefix: str,
    data: Optional[dict[str, Any]] = None,
) -> tuple[list[Path], list[str]]:
    import arviz as az

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_vars = _selected_vars(idata, var_names)
    files: list[Path] = []
    warnings: list[str] = []

    if plot_vars is not None:
        plot_specs = [
            (
                "trace.png",
                lambda: az.plot_trace(
                    idata, var_names=plot_vars, compact=True
                ),
            ),
            (
                "posterior.png",
                lambda: az.plot_posterior(idata, var_names=plot_vars),
            ),
            (
                "forest.png",
                lambda: az.plot_forest(
                    idata,
                    var_names=plot_vars,
                    combined=True,
                    hdi_prob=0.68,
                ),
            ),
        ]
        if len(plot_vars) > 1:
            plot_specs.append(
                (
                    "corner.png",
                    lambda: az.plot_pair(
                        idata,
                        var_names=plot_vars,
                        kind="kde",
                        marginals=True,
                    ),
                )
            )
        for filename, plot_func in plot_specs:
            path = plot_dir / _prefixed(file_prefix, filename)
            _write_plot(path, plot_func, warnings)
            if path.exists():
                files.append(path)

    if hasattr(idata, "sample_stats") and "energy" in idata.sample_stats:
        path = plot_dir / _prefixed(file_prefix, "energy.png")
        _write_plot(path, lambda: az.plot_energy(idata), warnings)
        if path.exists():
            files.append(path)

    if (
        hasattr(idata, "posterior_predictive")
        and hasattr(idata, "observed_data")
        and "y_scaled" in idata.posterior_predictive
        and "y_scaled" in idata.observed_data
    ):
        path = plot_dir / _prefixed(file_prefix, "ppc_y_scaled.png")
        _write_plot(
            path,
            lambda: az.plot_ppc(
                idata,
                data_pairs={"y_scaled": "y_scaled"},
                num_pp_samples=100,
            ),
            warnings,
        )
        if path.exists():
            files.append(path)

    if data is not None:
        path = plot_dir / _prefixed(file_prefix, "regression.png")
        _write_plot(
            path,
            lambda: _regression_plot(idata, **data),
            warnings,
        )
        if path.exists():
            files.append(path)

    return files, warnings


def write_report(
    idata: Any,
    output_dir: str | Path,
    *,
    var_names: Optional[Sequence[str]] = DEFAULT_REPORT_VARS,
    hdi_prob: float = 0.68,
    file_prefix: str = "",
    save_netcdf: bool = True,
    save_summary: bool = True,
    save_plots: bool = True,
    x: Any = None,
    y: Any = None,
    dy: Any = None,
    cov_x: Any = None,
) -> dict[str, Any]:
    """Write standard t-cup result artifacts to ``output_dir``.

    The generated artifacts are intentionally plain: an ArviZ NetCDF file,
    summary tables, a small Markdown report, and optional diagnostic plots.
    The input ``InferenceData`` is returned unchanged by the fitting routines;
    this helper only serializes a reusable report alongside it.
    """

    import arviz as az

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {"output_dir": output_dir}

    selected_vars = _selected_vars(idata, var_names)
    summary_text = ""

    if save_netcdf:
        path = output_dir / _prefixed(file_prefix, "inference_data.nc")
        idata.to_netcdf(path)
        artifacts["netcdf"] = path

    if save_summary and hasattr(idata, "posterior"):
        summary = az.summary(
            idata,
            var_names=selected_vars,
            hdi_prob=hdi_prob,
        )
        csv_path = output_dir / _prefixed(file_prefix, "summary.csv")
        txt_path = output_dir / _prefixed(file_prefix, "summary.txt")
        summary.to_csv(csv_path)
        summary_text = summary.to_string()
        txt_path.write_text(f"{summary_text}\n", encoding="utf-8")
        artifacts["summary_csv"] = csv_path
        artifacts["summary_txt"] = txt_path

    plot_files: list[Path] = []
    plot_warnings: list[str] = []
    if save_plots:
        data = None
        if x is not None and y is not None:
            data = {"x": x, "y": y, "dy": dy, "cov_x": cov_x}
        plot_files, plot_warnings = _write_plots(
            idata,
            output_dir,
            selected_vars,
            file_prefix,
            data=data,
        )
        artifacts["plots"] = plot_files

    report_path = output_dir / _prefixed(file_prefix, "report.md")
    lines = [
        "# t-cup report",
        "",
        "## Groups",
        "",
        *[f"- `{group}`" for group in idata.groups()],
        "",
        "## Posterior Variables",
        "",
        *(_dims_report(idata) or ["No posterior variables found."]),
        "",
        "## Files",
        "",
    ]

    for key, value in artifacts.items():
        if key == "output_dir":
            continue
        if isinstance(value, list):
            lines.extend(
                f"- `{item.relative_to(output_dir)}`" for item in value
            )
        else:
            lines.append(f"- `{value.relative_to(output_dir)}`")

    if summary_text:
        lines.extend(["", "## Summary", "", "```text", summary_text, "```"])

    if plot_warnings:
        warning_path = output_dir / _prefixed(file_prefix, "plot_warnings.txt")
        warning_path.write_text("\n".join(plot_warnings) + "\n")
        artifacts["plot_warnings"] = warning_path
        lines.extend(
            [
                "",
                "## Plot Warnings",
                "",
                *[f"- {warning}" for warning in plot_warnings],
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts["report"] = report_path
    return artifacts
