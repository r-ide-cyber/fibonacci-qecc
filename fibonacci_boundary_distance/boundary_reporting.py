"""Comparisons and figures for the finite Fibonacci boundary experiment.

All distances are measured from a window edge to a chain edge.  ``B_star`` is
an observed-grid crossing, not a fitted correlation length or evidence for an
infinite-system limit.  Missing quantum rates remain missing throughout.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


_CONDITION = [
    "geometry", "pair_id", "inflation_n", "N", "window_size", "erasure_size",
    "erasure_type",
]
_POSITION = [
    "boundary_side", "distance_mode", "boundary_distance",
    "boundary_distance_left", "boundary_distance_right", "window_start", "window_end",
]
_OPTIONAL_IDENTITY = ["model_id"]
_OPTIONAL_METADATA = [
    "fraction_requested", "boundary_fraction_actual", "open_support_is_singleton",
]
_METRICS = {"recoverability": "R_rec", "KL": "R_KL"}


def _conditions(frame: pd.DataFrame) -> list[str]:
    return _CONDITION + [c for c in _OPTIONAL_IDENTITY if c in frame.columns]


def _prepare(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    if frame.empty:
        for col in _CONDITION + _POSITION + ["R_rec", "R_KL", "number_of_patterns", "calculation_mode"]:
            if col not in frame:
                frame[col] = pd.Series(dtype="object")
        return frame
    required = _CONDITION + _POSITION + ["R_rec", "number_of_patterns", "calculation_mode"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")
    if "R_KL" not in frame:
        frame["R_KL"] = np.nan
    for col in ["R_rec", "R_KL", "boundary_distance", "boundary_distance_left", "boundary_distance_right"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    return frame


def _joined_modes(values: pd.Series) -> str:
    modes = sorted(set(values.dropna().astype(str)))
    return modes[0] if len(modes) == 1 else "mixed:" + "+".join(modes) if modes else "unknown"


def make_wide_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the requested wide table without losing per-type sample counts.

    ``number_of_patterns`` is the sum over erasure types, not a claim that the
    union of the two pattern sets contains that many distinct masks.  Per-type
    counts and exact/sampled labels are retained in separate columns.
    """
    frame = _prepare(summary)
    identity = [c for c in _conditions(frame) if c != "erasure_type"] + _POSITION
    identity += [c for c in _OPTIONAL_METADATA if c in frame.columns]
    outputs = [
        f"{metric}_{kind}" for kind in ["contiguous", "arbitrary"]
        for metric in ["R_rec", "R_KL", "number_of_patterns", "calculation_mode"]
    ]
    columns = identity + outputs + ["number_of_patterns", "calculation_mode"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    unknown = set(frame.erasure_type.unique()) - {"contiguous", "arbitrary"}
    if unknown:
        raise ValueError(f"Unknown erasure types: {sorted(unknown)}")
    if frame.duplicated(identity + ["erasure_type"]).any():
        raise ValueError("Summary contains duplicate condition/position/erasure-type rows.")
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(identity, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(identity, key))
        for kind in ["contiguous", "arbitrary"]:
            subset = group[group.erasure_type.eq(kind)]
            for metric in ["R_rec", "R_KL", "number_of_patterns", "calculation_mode"]:
                row[f"{metric}_{kind}"] = subset.iloc[0][metric] if len(subset) else np.nan
        row["number_of_patterns"] = pd.to_numeric(group.number_of_patterns).sum(min_count=1)
        row["calculation_mode"] = _joined_modes(group.calculation_mode)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _threshold_metric(group: pd.DataFrame, name: str, epsilon: float) -> dict[str, Any]:
    rate = _METRICS[name]
    center = f"{rate}_center"
    result: dict[str, Any] = {
        f"B_star_{name}": np.nan,
        f"B_star_sustained_grid_{name}": np.nan,
        f"status_{name}": "not_evaluated",
        f"sustained_grid_status_{name}": "not_evaluated",
        f"evaluated_grid_count_{name}": 0,
    }
    if group.empty:
        result[f"status_{name}"] = "no_admissible_distances"
        result[f"sustained_grid_status_{name}"] = "no_admissible_distances"
        return result
    if group[rate].isna().all():
        return result
    if group[center].isna().all():
        result[f"status_{name}"] = "center_missing"
        result[f"sustained_grid_status_{name}"] = "center_missing"
        return result
    ordered = group.sort_values("boundary_distance")
    finite = ordered[[rate, center]].notna().all(axis=1)
    close = finite & (ordered[rate] - ordered[center]).abs().lt(epsilon)
    result[f"evaluated_grid_count_{name}"] = int(finite.sum())
    if close.any():
        result[f"B_star_{name}"] = float(ordered.loc[close, "boundary_distance"].min())
        result[f"status_{name}"] = "observed_crossing"
    else:
        result[f"status_{name}"] = "no_crossing_on_observed_grid"
    sustained = np.logical_and.accumulate(close.to_numpy(dtype=bool)[::-1])[::-1]
    if sustained.any():
        result[f"B_star_sustained_grid_{name}"] = float(ordered.loc[sustained, "boundary_distance"].min())
        result[f"sustained_grid_status_{name}"] = "sustained_on_observed_grid_only"
    else:
        result[f"sustained_grid_status_{name}"] = "no_sustained_crossing_on_observed_grid"
    return result


def derive_comparisons(
    summary: pd.DataFrame, epsilon: float = 0.01
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return center differences, open-chain thresholds, and placement mirrors.

    Center differences use ``R_center - R(B)``.  Thresholds use the strict
    criterion ``abs(R_center - R(B)) < epsilon`` on fixed-site samples.  Points
    beyond the midpoint are excluded, because their named boundary is no
    longer the nearest boundary.  The sustained variant additionally requires
    every later *observed* point to meet the criterion; unobserved distances
    are never certified.  The center row itself is not added to the scan grid.

    Mirror comparisons translate the same relative masks to left and right
    placements.  They do not reflect masks, and asymmetry is conditional on
    the chosen ensemble and (when sampled) the sampled masks.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive.")
    frame = _prepare(summary)
    condition = _conditions(frame)
    delta_extra = [
        "R_rec_center", "R_KL_center", "Delta_boundary_recoverability", "Delta_boundary_KL",
        "center_calculation_mode", "center_number_of_patterns", "has_center_reference",
    ]
    threshold_columns = condition + [
        "boundary_side", "distance_mode", "epsilon", "R_center", "R_center_recoverability",
        "R_center_KL", "B_star_recoverability", "B_star_KL",
        "B_star_sustained_grid_recoverability", "B_star_sustained_grid_KL",
        "status_recoverability", "status_KL", "sustained_grid_status_recoverability",
        "sustained_grid_status_KL", "evaluated_grid_count_recoverability", "evaluated_grid_count_KL",
        "observed_grid_count", "observed_B_min", "observed_B_max",
        "excluded_beyond_midpoint_count", "calculation_mode", "interpretation",
    ]
    mirror_columns = condition + [
        "distance_mode", "boundary_distance", "R_rec_left", "R_rec_right", "R_KL_left", "R_KL_right",
        "Delta_left_minus_right_recoverability", "Delta_left_minus_right_KL",
        "asymmetry_recoverability", "asymmetry_KL", "comparison_basis", "interpretation",
    ]
    if frame.empty:
        return (
            pd.DataFrame(columns=list(frame.columns) + delta_extra),
            pd.DataFrame(columns=threshold_columns),
            pd.DataFrame(columns=mirror_columns),
        )
    centers = frame[frame.boundary_side.eq("center")]
    if centers.duplicated(condition).any():
        raise ValueError("Multiple center references exist for the same model/condition.")
    centers = centers[condition + ["R_rec", "R_KL", "calculation_mode", "number_of_patterns"]].rename(
        columns={"R_rec": "R_rec_center", "R_KL": "R_KL_center",
                 "calculation_mode": "center_calculation_mode", "number_of_patterns": "center_number_of_patterns"}
    )
    delta = frame.merge(centers, how="left", on=condition, validate="many_to_one", indicator="_center_merge")
    delta["has_center_reference"] = delta.pop("_center_merge").eq("both")
    for name, metric in _METRICS.items():
        delta[f"Delta_boundary_{name}"] = delta[f"{metric}_center"] - delta[metric]

    fixed = delta[delta.geometry.eq("open") & delta.distance_mode.eq("sites") & delta.boundary_side.isin(["left", "right"])]
    threshold_rows: list[dict[str, Any]] = []
    threshold_identity = condition + ["boundary_side"]
    for key, all_points in fixed.groupby(threshold_identity, dropna=False, sort=False):
        row = dict(zip(threshold_identity, key))
        # The named boundary must be a nearest boundary, including a midpoint tie.
        if row["boundary_side"] == "left":
            admissible = all_points.boundary_distance_left.le(all_points.boundary_distance_right)
        else:
            admissible = all_points.boundary_distance_right.le(all_points.boundary_distance_left)
        points = all_points.loc[admissible].copy()
        if points.duplicated("boundary_distance").any():
            raise ValueError("Duplicate site distances in a threshold condition.")
        row.update({
            "distance_mode": "sites", "epsilon": float(epsilon),
            "R_center": all_points.R_rec_center.iloc[0],
            "R_center_recoverability": all_points.R_rec_center.iloc[0],
            "R_center_KL": all_points.R_KL_center.iloc[0],
            "observed_grid_count": len(points),
            "observed_B_min": points.boundary_distance.min(),
            "observed_B_max": points.boundary_distance.max(),
            "excluded_beyond_midpoint_count": int((~admissible).sum()),
            "calculation_mode": _joined_modes(points.calculation_mode),
            "interpretation": "observed_grid_only_not_a_bulk_limit",
        })
        if "open_support_is_singleton" in all_points and all_points.open_support_is_singleton.fillna(False).all():
            row["interpretation"] = "trivial_singleton_support_not_boundary_evidence"
        for name in _METRICS:
            row.update(_threshold_metric(points, name, epsilon))
        threshold_rows.append(row)
    thresholds = pd.DataFrame(threshold_rows, columns=threshold_columns)

    mirror_identity = condition + ["distance_mode", "boundary_distance"]
    if "fraction_requested" in frame:
        mirror_identity.append("fraction_requested")
    sides = frame[frame.geometry.eq("open") & frame.boundary_side.isin(["left", "right"])]
    left = sides[sides.boundary_side.eq("left")]
    right = sides[sides.boundary_side.eq("right")]
    keep = ["R_rec", "R_KL", "calculation_mode", "number_of_patterns", "window_start", "window_end"]
    mirror = left[mirror_identity + keep].merge(
        right[mirror_identity + keep], on=mirror_identity, suffixes=("_left", "_right"), validate="one_to_one"
    )
    for name, metric in _METRICS.items():
        difference = mirror[f"{metric}_left"] - mirror[f"{metric}_right"]
        mirror[f"Delta_left_minus_right_{name}"] = difference
        mirror[f"asymmetry_{name}"] = np.where(
            difference.isna(), "not_evaluated",
            np.where(difference.abs().ge(epsilon), "conditional_placement_asymmetry", "within_epsilon"),
        )
    mirror["epsilon"] = float(epsilon)
    mirror["comparison_basis"] = "same relative patterns translated; patterns are not reflected"
    mirror["interpretation"] = "conditional_on_model_and_pattern_ensemble; sampled_differences_are_not_significance_tests"
    return delta, thresholds, mirror


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"


def _singleton_note(group: pd.DataFrame) -> str:
    singleton = (
        "open_support_is_singleton" in group
        and group.geometry.eq("open").all()
        and group.open_support_is_singleton.fillna(False).all()
    )
    if singleton or (
        "interpretation" in group
        and group.interpretation.eq("trivial_singleton_support_not_boundary_evidence").all()
    ):
        return "\nSingleton supports: trivial R_rec=1; not boundary evidence"
    return ""


def plot_boundary(
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    threshold: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> list[Any]:
    """Create separate scientific figures for geometry, metric and scan mode.

    Each W,m condition has distinct contiguous/arbitrary panels.  Color
    identifies inflation n and line style identifies the boundary side.
    Dashed horizontal lines give each n's matching center reference.  No KL
    figure is created if its rates are uncomputed.  Returns open Figure
    objects so the caller can display or close them.
    """
    import matplotlib.pyplot as plt

    frame = _prepare(summary)
    if frame.empty:
        return []
    destination = Path(output_dir) if output_dir is not None else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    figures: list[Any] = []
    group_columns = ["geometry", "pair_id", "window_size", "erasure_size"] + [c for c in _OPTIONAL_IDENTITY if c in frame]
    rc = {"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
          "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 200,
          "figure.max_open_warning": 0}

    def finish(fig: Any, name: str) -> None:
        figures.append(fig)
        if destination:
            fig.savefig(destination / f"{name}.png", bbox_inches="tight")

    with plt.rc_context(rc):
        for key, group in frame.groupby(group_columns, dropna=False, sort=False):
            ids = dict(zip(group_columns, key))
            tag = "_".join(f"{_slug(k)}{_slug(v)}" for k, v in ids.items())
            title = f"{ids['geometry'].capitalize()} | W={ids['window_size']}, m={ids['erasure_size']} | {ids['pair_id']}"
            if "model_id" in ids:
                title += f" | {ids['model_id']}"
            ns = sorted(group.inflation_n.unique())
            palette = plt.get_cmap("tab10")
            colors = {n: palette(i % 10) for i, n in enumerate(ns)}
            for mode in ["sites", "fraction"]:
                scan = group[group.distance_mode.eq(mode) & ~group.boundary_side.eq("center")]
                if scan.empty:
                    continue
                for name, metric in _METRICS.items():
                    if scan[metric].notna().sum() == 0:
                        continue
                    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True, squeeze=False, constrained_layout=True)
                    fig.suptitle(title + (" | fixed fraction" if mode == "fraction" else " | fixed site distance") + _singleton_note(group))
                    for ax, kind in zip(axes[0], ["contiguous", "arbitrary"]):
                        kind_scan = scan[scan.erasure_type.eq(kind)]
                        for (n, side), line in kind_scan.groupby(["inflation_n", "boundary_side"], sort=True):
                            line = line.sort_values("boundary_distance")
                            x = line.boundary_distance / line.N if mode == "fraction" else line.boundary_distance
                            N = line.N.iloc[0]
                            label = f"n={n}, N={N}; {side}; {_joined_modes(line.calculation_mode)}"
                            ax.plot(x, line[metric], color=colors[n], linestyle="-" if side == "left" else ":", marker="o" if side == "left" else "s", markersize=4, label=label)
                        for n in kind_scan.inflation_n.unique():
                            ref = group[group.inflation_n.eq(n) & group.erasure_type.eq(kind) & group.boundary_side.eq("center")][metric].dropna()
                            if len(ref):
                                ax.axhline(float(ref.iloc[0]), color=colors[n], linestyle="--", linewidth=1.1, label=f"n={n} center")
                        ax.set_title(kind.capitalize())
                        xlabel = "Realized B/N" if mode == "fraction" else "B (sites from selected boundary)"
                        if ids["geometry"] == "ring":
                            xlabel = "Window placement parameter B/N (periodic ring)" if mode == "fraction" else "Window placement parameter B (periodic ring)"
                        ax.set_xlabel(xlabel)
                        ax.set_ylim(-0.035, 1.035)
                        ax.grid(alpha=0.22)
                        if ax.lines:
                            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=7, framealpha=0.8)
                    axes[0, 0].set_ylabel("Classical recoverability rate" if metric == "R_rec" else "KL correctability rate")
                    finish(fig, f"boundary_{name}_{mode}_{tag}")

            if not delta.empty:
                selected = delta.copy()
                for col, value in ids.items():
                    selected = selected[selected[col].isna()] if pd.isna(value) else selected[selected[col].eq(value)]
                selected = selected[selected.distance_mode.eq("sites") & ~selected.boundary_side.eq("center")]
                for name in _METRICS:
                    column = f"Delta_boundary_{name}"
                    if column not in selected or selected[column].notna().sum() == 0:
                        continue
                    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True, squeeze=False, constrained_layout=True)
                    fig.suptitle(title + " | difference from center" + _singleton_note(group))
                    for ax, kind in zip(axes[0], ["contiguous", "arbitrary"]):
                        for (n, side), line in selected[selected.erasure_type.eq(kind)].groupby(["inflation_n", "boundary_side"], sort=True):
                            line = line.sort_values("boundary_distance")
                            ax.plot(line.boundary_distance, line[column], color=colors[n], linestyle="-" if side == "left" else ":", marker="o" if side == "left" else "s", markersize=4, label=f"n={n}; {side}; {_joined_modes(line.calculation_mode)}")
                        ax.axhline(0, color="0.4", linewidth=1, linestyle="--")
                        ax.set_title(kind.capitalize())
                        ax.set_xlabel("B (sites from selected boundary)" if ids["geometry"] == "open" else "Window placement parameter B (periodic ring)")
                        ax.grid(alpha=0.22)
                        if len(ax.lines) > 1:
                            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=7, framealpha=0.8)
                    axes[0, 0].set_ylabel(f"Center rate − local rate ({name})")
                    finish(fig, f"boundary_delta_{name}_{tag}")

        if not threshold.empty:
            threshold_groups = [c for c in group_columns if c in threshold]
            for key, group in threshold.groupby(threshold_groups, dropna=False, sort=False):
                ids = dict(zip(threshold_groups, key if isinstance(key, tuple) else (key,)))
                tag = "_".join(f"{_slug(k)}{_slug(v)}" for k, v in ids.items())
                for name in _METRICS:
                    metric = f"B_star_{name}"
                    sustained = f"B_star_sustained_grid_{name}"
                    # A missing KL definition must not produce an empty quantum plot.
                    if group[metric].notna().sum() == 0 and group[sustained].notna().sum() == 0:
                        continue
                    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True, squeeze=False, constrained_layout=True)
                    threshold_title = f"Open-chain observed thresholds | W={ids['window_size']}, m={ids['erasure_size']} | {ids['pair_id']}"
                    if "model_id" in ids:
                        threshold_title += f" | {ids['model_id']}"
                    fig.suptitle(threshold_title + _singleton_note(group))
                    for ax, kind in zip(axes[0], ["contiguous", "arbitrary"]):
                        for i, (side, line) in enumerate(group[group.erasure_type.eq(kind)].groupby("boundary_side", sort=True)):
                            line = line.sort_values("N")
                            color = plt.get_cmap("tab10")(i)
                            label = f"{side}; {_joined_modes(line.calculation_mode)}"
                            ax.plot(line.N, line[metric], color=color, marker="o", linestyle="-", label=label + "; first crossing")
                            ax.plot(line.N, line[sustained], color=color, marker="s", linestyle="--", label=label + "; sustained grid")
                        ax.set_title(kind.capitalize())
                        ax.set_xlabel("N (inflated chain length)")
                        ax.grid(alpha=0.22)
                        if ax.lines:
                            ax.legend(loc="best", framealpha=0.8)
                    axes[0, 0].set_ylabel(f"B* (sites; {name})")
                    finish(fig, f"boundary_threshold_{name}_{tag}")
    return figures
