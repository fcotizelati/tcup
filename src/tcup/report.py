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


def _write_plots(
    idata: Any,
    output_dir: Path,
    var_names: Optional[Sequence[str]],
    file_prefix: str,
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
        plot_files, plot_warnings = _write_plots(
            idata,
            output_dir,
            selected_vars,
            file_prefix,
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
