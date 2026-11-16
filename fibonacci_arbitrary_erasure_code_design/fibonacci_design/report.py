"""Plots for the staged seed-pair search, with experimental conditions explicit.

No plotting occurs on import. ``make_figures`` returns Matplotlib figures and
optionally saves PNGs. Curve panels retain separate N/W/m conditions; feature
plots use one common training condition at the screening inflation.
"""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


FIGURE_NAMES = (
    "design_pareto",
    "design_score_vs_KL",
    "d_cross_vs_KL",
    "collision_vs_KL",
    "sequence_feature_vs_KL",
    "leakage_vs_KL",
    "baseline_vs_candidates",
    "candidate_performance_vs_m",
    "candidate_performance_vs_N",
    "candidate_improvement_vs_N",
    "improvement_vs_m",
)

_PERFORMANCE_COLUMNS = (
    "candidate_id", "inflation_n", "N", "window_size", "erasure_size",
    "sample_split", "R_KL_arbitrary", "R_KL_contiguous",
)
_CONDITION_COLUMNS = ("inflation_n", "N", "window_size", "erasure_size", "sample_split")


def _frame(tables, name):
    source = tables.get(name)
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.DataFrame()
    if "candidate_id" in frame:
        frame["candidate_id"] = frame["candidate_id"].astype(str)
    return frame


def _flags(frame, column):
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(lambda value: str(value).strip().lower() in {"true", "1", "1.0"})


def _numeric(frame, columns):
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _performance(frame):
    if frame.empty or not set(_PERFORMANCE_COLUMNS).issubset(frame.columns):
        return pd.DataFrame(columns=_PERFORMANCE_COLUMNS)
    frame = _numeric(frame, ("inflation_n", "N", "window_size", "erasure_size",
                             "R_KL_arbitrary", "R_KL_contiguous",
                             "Delta_R_arbitrary", "Delta_R_contiguous"))
    frame = frame.dropna(subset=list(_PERFORMANCE_COLUMNS))
    # Resume files may contain the same completed calculation more than once.
    return frame.drop_duplicates(["candidate_id", *_CONDITION_COLUMNS], keep="last")


def _empty(title, reason):
    figure, axis = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.text(0.5, 0.5, reason, transform=axis.transAxes, ha="center", va="center",
              wrap=True, color="#64748b")
    axis.set_axis_off()
    return figure


def _integer_label(value):
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _condition_label(frame):
    row = frame.iloc[0]
    return ", ".join((
        f"{row['sample_split']}",
        f"n={_integer_label(row['inflation_n'])}",
        f"N={_integer_label(row['N'])}",
        f"W={_integer_label(row['window_size'])}",
        f"m={_integer_label(row['erasure_size'])}",
    ))


def _representative(frame, *, split=None, inflation=None):
    """Choose one shared condition: selected/min n, max N, max W, max m."""
    selected = frame
    if selected.empty:
        return selected
    if split is not None:
        selected = selected[selected["sample_split"] == split]
    if selected.empty:
        return selected
    if inflation is None:
        inflation = selected["inflation_n"].min()
    selected = selected[selected["inflation_n"] == inflation]
    for column in ("N", "window_size", "erasure_size"):
        if selected.empty:
            break
        selected = selected[selected[column] == selected[column].max()]
    return selected


def _screening_inflation(screening, performance):
    for column in ("screening_inflation_n", "inflation_n"):
        if column in screening:
            values = pd.to_numeric(screening[column], errors="coerce").dropna()
            if not values.empty:
                return values.iloc[0]
    train = performance[performance["sample_split"] == "train"]
    return train["inflation_n"].min() if not train.empty else None


def _candidate_order(screening, final, performance, baseline_ids):
    available = set(performance["candidate_id"])
    ordered = []
    if not final.empty and "candidate_id" in final:
        final = final.copy()
        final["_passed"] = _flags(final, "validation_pass")
        if "rank" in final:
            final["rank"] = pd.to_numeric(final["rank"], errors="coerce")
            final = final.sort_values(["_passed", "rank"], ascending=[False, True])
        else:
            final = final.sort_values("_passed", ascending=False, kind="stable")
        ordered.extend(final["candidate_id"])
    if not screening.empty and "candidate_id" in screening:
        if "design_score" in screening:
            screening = _numeric(screening.copy(), ["design_score"]).sort_values("design_score", ascending=False)
        ordered.extend(screening["candidate_id"])
    ordered.extend(sorted(available))
    candidates = list(dict.fromkeys(candidate for candidate in ordered
                                    if candidate in available and candidate not in baseline_ids))[:3]
    return sorted(baseline_ids & available) + candidates


def _short_id(candidate):
    return candidate if len(candidate) <= 27 else candidate[:16] + "…" + candidate[-8:]


def _label(candidate, baseline_ids):
    return f"Baseline: {_short_id(candidate)}" if candidate in baseline_ids else _short_id(candidate)


def _style_axis(axis):
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)


def _scatter(axis, frame, x, y, baseline_ids, *, pareto=False):
    frame = _numeric(frame.copy(), [x, y]).dropna(subset=[x, y])
    if frame.empty:
        axis.text(0.5, 0.5, "No finite matched values", ha="center", va="center", transform=axis.transAxes)
        return
    baseline = frame["candidate_id"].isin(baseline_ids)
    front = _flags(frame, "pareto_optimal") & ~baseline if pareto else pd.Series(False, index=frame.index)
    for selection, label, color, marker, size in (
        (~baseline & ~front, "Candidates", "#2563eb", "o", 42),
        (front, "Screening Pareto candidates", "#0f766e", "D", 54),
        (baseline, "Baseline", "#b45309", "*", 190),
    ):
        if selection.any():
            points = frame[selection]
            axis.scatter(points[x], points[y], label=label, color=color, marker=marker,
                         s=size, alpha=0.9, edgecolor="white", linewidth=0.6, zorder=3)
    # Coincident scientific values must stay coincident; aggregate their labels.
    if len(frame) <= 12:
        for (xv,yv), group in frame.groupby([x,y],sort=False):
            includes_base=group.candidate_id.isin(baseline_ids).any()
            label=(f"{len(group)} pairs" + (" (incl. baseline)" if includes_base else "")) if len(group)>1 else ("Baseline" if includes_base else "")
            if label:
                axis.annotate(label, (xv,yv), xytext=(7,8), textcoords="offset points", fontsize=8)
    axis.legend(frameon=False, fontsize=8, loc="best")
    _style_axis(axis)


def _panel_grid(count):
    columns = min(3, count)
    rows = ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 3.8 * rows),
                                squeeze=False, layout="constrained")
    for axis in axes.flat[count:]:
        axis.set_axis_off()
    return figure, list(axes.flat[:count])


def _curve_figure(frame, selected, baseline_ids, *, by_size=False, improvement=False):
    title = ("Change from baseline vs physical size" if improvement and by_size else
             "Change from baseline vs erased sites" if improvement else
             "Arbitrary-erasure KL rate vs physical size" if by_size else
             "Arbitrary-erasure KL rate vs erased sites")
    if frame.empty or not selected:
        return _empty(title, "No evaluated candidate conditions are available.")
    split = "validation" if (frame["sample_split"] == "validation").any() else "train"
    frame = frame[(frame["sample_split"] == split) & frame["candidate_id"].isin(selected)].copy()
    if improvement:
        required = {"Delta_R_arbitrary", "Delta_R_contiguous"}
        if not required.issubset(frame.columns):
            return _empty(title, "Baseline improvement columns are unavailable.")
        frame = frame[~frame["candidate_id"].isin(baseline_ids)]
    if frame.empty:
        return _empty(title, f"No candidate rows for {split}.")
    if by_size:
        frame["_window_policy"] = ["W=N (full ring)" if w == n else f"W={_integer_label(w)} (fixed window)"
                                    for w, n in zip(frame["window_size"], frame["N"])]
        groups = list(frame.groupby(["_window_policy", "erasure_size"], sort=True))
    else:
        groups = list(frame.groupby(["inflation_n", "N", "window_size"], sort=True))
    figure, axes = _panel_grid(len(groups))
    palette = plt.get_cmap("tab10")
    colors = {candidate: ("#b45309" if candidate in baseline_ids else palette(i % 10))
              for i, candidate in enumerate(selected)}
    for axis, (condition, group) in zip(axes, groups):
        if by_size:
            axis.set_title(f"{condition[0]}, m={_integer_label(condition[1])}", fontsize=10)
            x = "N"
        else:
            n, size, window = condition
            axis.set_title(f"n={_integer_label(n)}, N={_integer_label(size)}, W={_integer_label(window)}", fontsize=10)
            x = "erasure_size"
        for candidate in selected:
            subset = group[group["candidate_id"] == candidate].sort_values(x)
            if subset.empty:
                continue
            metrics = (("Delta_R_arbitrary", "-"), ("Delta_R_contiguous", "--")) if improvement else (("R_KL_arbitrary", "-"),)
            for metric, linestyle in metrics:
                axis.plot(subset[x], subset[metric], color=colors[candidate], linestyle=linestyle,
                          marker="o", markersize=3.5, linewidth=1.6,
                          label=_label(candidate, baseline_ids) if linestyle == "-" else "_nolegend_")
        if improvement:
            axis.axhline(0, color="#64748b", linewidth=0.9)
            axis.set_ylabel("Candidate rate − baseline rate")
        else:
            axis.set_ylim(-0.035, 1.055)
            axis.set_ylabel("Arbitrary-erasure KL correction rate")
        axis.set_xlabel("Physical sites N" if by_size else "Erased sites m")
        axis.legend(frameon=False, fontsize=8, loc="best")
        _style_axis(axis)
    suffix = "; solid: arbitrary, dashed: contiguous" if improvement else ""
    comparison="up to 3 ranked candidates; baseline = zero" if improvement else "baseline + up to 3 ranked candidates"
    figure.suptitle(f"{title}\n{split}; {comparison}{suffix}", fontsize=12)
    return figure


def make_figures(tables: dict[str, pd.DataFrame], output_dir, save=True) -> dict[str, plt.Figure]:
    """Return all eleven named figures, optionally saving ``<name>.png`` at 140 dpi.

    Empty/missing tables produce informative placeholder figures. Screening
    feature comparisons and baseline bars use training data at the screening
    inflation, the largest available N/W, and then the largest available m.
    Every plot states that condition. Curves prefer validation and show all
    condition panels for the baseline and at most three ranked candidates.
    Fixed windows and full rings use separate N-curve panels. Latest duplicate
    performance rows are retained, and no averaging across N/W/m is performed.
    ``aggregate`` is accepted for compatibility but is not substituted for
    measurements at explicit conditions. Returned figures remain open.
    """
    screening = _frame(tables, "screening")
    performance = _performance(_frame(tables, "performance"))
    leakage = _frame(tables, "leakage")
    final = _frame(tables, "final")
    baseline_ids = set()
    for frame in (screening, performance):
        if "candidate_id" in frame:
            baseline_ids.update(frame.loc[_flags(frame, "is_baseline"), "candidate_id"])
    selected = _candidate_order(screening, final, performance, baseline_ids)
    inflation = _screening_inflation(screening, performance)
    representative = _representative(performance, split="train", inflation=inflation)
    condition = _condition_label(representative) if not representative.empty else "No matching training condition"
    screening = screening.drop_duplicates("candidate_id", keep="last") if "candidate_id" in screening else screening
    joined = representative.merge(screening, on="candidate_id", how="inner", suffixes=("", "_feature")) if "candidate_id" in screening else pd.DataFrame()
    figures = {}

    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"}):
        if representative.empty:
            figures["design_pareto"] = _empty("Measured correction-rate trade-off", condition)
        else:
            fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
            plot_data = representative
            if "pareto_optimal" in screening:
                plot_data = representative.merge(screening[["candidate_id", "pareto_optimal"]], on="candidate_id", how="left")
            _scatter(ax, plot_data, "R_KL_arbitrary", "R_KL_contiguous", baseline_ids, pareto=True)
            ax.set(xlabel="Arbitrary-erasure KL correction rate", ylabel="Contiguous-erasure KL correction rate",
                   xlim=(-0.035, 1.055), ylim=(-0.035, 1.055), title=f"Measured correction-rate trade-off\n{condition}")
            figures["design_pareto"] = fig

        for name, column, xlabel in (
            ("design_score_vs_KL", "design_score", "Screening design score"),
            ("d_cross_vs_KL", "d_cross", "Minimum cross-orbit Hamming distance"),
            ("collision_vs_KL", "expected_cross_collision_arbitrary", "Expected arbitrary cross-collision count (screening)"),
        ):
            if joined.empty or column not in joined:
                figures[name] = _empty(xlabel + " vs KL", "Matching screening features and training KL results are unavailable.")
                continue
            fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
            _scatter(ax, joined, column, "R_KL_arbitrary", baseline_ids)
            ax.set(xlabel=xlabel, ylabel="Arbitrary-erasure KL correction rate", ylim=(-0.035, 1.055),
                   title=f"Screening feature vs measured KL rate\n{condition}")
            figures[name] = fig

        sequence_columns = [("long_range_C2_mismatch", "Long-range C2 mismatch"), ("C3_mismatch", "C3 mismatch")]
        available_sequence = [(column, label) for column, label in sequence_columns if column in joined]
        if joined.empty or not available_sequence:
            figures["sequence_feature_vs_KL"] = _empty("Sequence mismatch vs KL", "Matching sequence features and training KL results are unavailable.")
        else:
            fig, axes = plt.subplots(1, len(available_sequence), figsize=(6.2 * len(available_sequence), 4.8),
                                      squeeze=False, layout="constrained")
            for ax, (column, label) in zip(axes.flat, available_sequence):
                _scatter(ax, joined, column, "R_KL_arbitrary", baseline_ids)
                ax.set(xlabel=label, ylabel="Arbitrary-erasure KL correction rate", ylim=(-0.035, 1.055))
            fig.suptitle(f"Screening sequence features vs measured KL rate\n{condition}")
            figures["sequence_feature_vs_KL"] = fig

        leak_keys = ["candidate_id", *_CONDITION_COLUMNS]
        if not leakage.empty and set(leak_keys + ["max_I_RE"]).issubset(leakage.columns):
            leakage = _numeric(leakage, ("inflation_n", "N", "window_size", "erasure_size", "max_I_RE"))
            leakage = leakage.drop_duplicates(leak_keys, keep="last")
            matched = performance.merge(leakage[leak_keys + ["max_I_RE"]], on=leak_keys, how="inner")
            leak_split = "validation" if (matched["sample_split"] == "validation").any() else "train"
            matched = _representative(matched, split=leak_split)
        else:
            matched = pd.DataFrame()
        if matched.empty:
            figures["leakage_vs_KL"] = _empty("Worst observed leakage vs KL", "No leakage and KL results at the same N/W/m/split are available.")
        else:
            fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
            _scatter(ax, matched, "max_I_RE", "R_KL_arbitrary", baseline_ids)
            ax.set(xlabel="Maximum observed I(R:E) [bits]", ylabel="Arbitrary-erasure KL correction rate",
                   ylim=(-0.035, 1.055), title=f"Worst observed leakage vs measured KL rate\n{_condition_label(matched)}")
            figures["leakage_vs_KL"] = fig

        bars = representative[representative["candidate_id"].isin(selected)]
        if bars.empty:
            figures["baseline_vs_candidates"] = _empty("Baseline and ranked candidates", "No matching training results are available.")
        else:
            bars = bars.set_index("candidate_id").reindex([candidate for candidate in selected if candidate in set(bars["candidate_id"])]).reset_index()
            fig, ax = plt.subplots(figsize=(max(7.2, len(bars) * 1.7), 5), layout="constrained")
            x = list(range(len(bars)))
            ax.bar([position - 0.19 for position in x], bars["R_KL_arbitrary"], width=0.36, label="Arbitrary", color="#2563eb")
            ax.bar([position + 0.19 for position in x], bars["R_KL_contiguous"], width=0.36, label="Contiguous", color="#0f766e")
            ax.set_xticks(x, [_label(candidate, baseline_ids) for candidate in bars["candidate_id"]], rotation=16, ha="right")
            ax.set(ylabel="KL correction rate", ylim=(0, 1.075), title=f"Baseline and up to 3 ranked candidates\n{condition}")
            ax.legend(frameon=False)
            _style_axis(ax)
            figures["baseline_vs_candidates"] = fig

        figures["candidate_performance_vs_m"] = _curve_figure(performance, selected, baseline_ids)
        figures["candidate_performance_vs_N"] = _curve_figure(performance, selected, baseline_ids, by_size=True)
        figures["candidate_improvement_vs_N"] = _curve_figure(performance, selected, baseline_ids, by_size=True, improvement=True)
        figures["improvement_vs_m"] = _curve_figure(performance, selected, baseline_ids, improvement=True)

        if save:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            for name, figure in figures.items():
                figure.savefig(destination / f"{name}.png", dpi=140, bbox_inches="tight")
    return figures
