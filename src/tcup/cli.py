"""Command line interface for fitting t-cup models from CSV files."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from . import tcup
from .report import ALL_PLOT_KINDS, DEFAULT_REPORT_VARS


def _split_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",")]
    return [column for column in columns if column]


def _read_columns(data: np.ndarray, columns: list[str]) -> np.ndarray:
    missing = [column for column in columns if column not in data.dtype.names]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    values = np.column_stack([data[column] for column in columns])
    if values.shape[1] == 1:
        return values[:, 0]
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a t-cup robust Bayesian regression from a CSV file."
    )
    parser.add_argument(
        "input", type=Path, help="Input CSV file with a header"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where report artifacts will be written",
    )
    parser.add_argument(
        "--x",
        default="x",
        help="x column name, or comma-separated names for multidimensional x",
    )
    parser.add_argument("--y", default="y", help="y column name")
    parser.add_argument(
        "--dx",
        default="dx",
        help="dx column name, or comma-separated names matching --x",
    )
    parser.add_argument("--dy", default="dy", help="dy column name")
    parser.add_argument(
        "--backend",
        choices=["numpyro", "stan"],
        default="numpyro",
        help="Fitting backend",
    )
    parser.add_argument(
        "--model",
        choices=["tcup", "ncup", "fixed"],
        default="tcup",
        help="Regression model variant",
    )
    parser.add_argument(
        "--shape-param",
        type=float,
        default=None,
        help="Fixed Student-t shape parameter for --model fixed",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--chain-method", default=None)
    parser.add_argument("--prior-samples", type=int, default=1000)
    parser.add_argument("--x-prior-components", type=int, default=None)
    parser.add_argument("--hdi-prob", type=float, default=0.68)
    parser.add_argument(
        "--report-var",
        action="append",
        dest="report_vars",
        help=(
            "Posterior variable to include in summaries and diagnostics. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--plot",
        action="append",
        dest="plot_kinds",
        choices=[*ALL_PLOT_KINDS, "all", "diagnostics"],
        help=(
            "Plot to write. Defaults to science-facing regression, corner, "
            "and parameter plots. Pass multiple times, or use diagnostics/all."
        ),
    )
    parser.add_argument(
        "--diagnostic-plots",
        action="store_true",
        help="Also write sampler and scaled posterior-predictive diagnostics.",
    )
    parser.add_argument("--no-netcdf", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data = np.genfromtxt(args.input, delimiter=",", names=True)
    if data.dtype.names is None:
        parser.error("Input CSV must include a header row")

    x_columns = _split_columns(args.x)
    dx_columns = _split_columns(args.dx)
    if len(dx_columns) not in {1, len(x_columns)}:
        parser.error("--dx must contain one column or match the --x columns")

    sampler_kwargs = {
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "prior_samples": args.prior_samples,
    }
    if args.chain_method is not None:
        sampler_kwargs["chain_method"] = args.chain_method

    report_vars = args.report_vars
    if report_vars is None:
        report_vars = list(DEFAULT_REPORT_VARS)
    plot_kinds = args.plot_kinds
    if args.diagnostic_plots:
        plot_kinds = [*(plot_kinds or []), "diagnostics"]

    tcup(
        x=_read_columns(data, x_columns),
        y=_read_columns(data, [args.y]),
        dx=_read_columns(data, dx_columns),
        dy=_read_columns(data, [args.dy]),
        seed=args.seed,
        backend=args.backend,
        model=args.model,
        shape_param=args.shape_param,
        x_prior_components=args.x_prior_components,
        output_dir=args.output_dir,
        report_kwargs={
            "var_names": report_vars,
            "hdi_prob": args.hdi_prob,
            "save_netcdf": not args.no_netcdf,
            "save_summary": not args.no_summary,
            "save_plots": not args.no_plots,
            "plot_kinds": plot_kinds,
        },
        **sampler_kwargs,
    )

    print(f"Wrote t-cup report to {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
