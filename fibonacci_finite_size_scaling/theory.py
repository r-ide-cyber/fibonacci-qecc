"""Li–Boyle Appendix E の有限 Fibonacci 符号の保証長と適用条件。"""
from __future__ import annotations

from numbers import Integral

import legacy_functions as legacy


PAPER_URL = "https://arxiv.org/abs/2311.13040v2"
PAPER_PDF_URL = "https://arxiv.org/pdf/2311.13040v2"
PAPER_REFERENCE = "Li & Boyle, arXiv:2311.13040v2, Appendix E.1, Eq. (E1), E.3.b Proposition 3"


def fibonacci(n):
    """原論文の添字: f_0=f_1=1, f_n=f_(n-1)+f_(n-2)。"""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    a, b = 1, 1
    for _ in range(int(n)):
        a, b = b, a + b
    return a


def L_theory(n):
    """式 f_n+1 の値。n=0 での保証はしない（適用条件は別途判定）。"""
    return fibonacci(n) + 1


def has_self_equivalent_adjacent_swap(seed):
    """ring の 01/10 交換で同じ巡回同値類へ戻る場合を検出する。"""
    legacy.validate_bitstring(seed, "seed")
    canonical = legacy.canonical_rotation(seed)
    for left in range(len(seed)):
        right = (left + 1) % len(seed)
        if seed[left] == seed[right]:
            continue
        bits = list(seed)
        bits[left], bits[right] = bits[right], bits[left]
        if legacy.canonical_rotation("".join(bits)) == canonical:
            return True
    return False


def paper_seed_conditions(seed_0, seed_1):
    """ユーザーの pair 条件を変更せず、原論文の追加条件も診断する。"""
    for label, seed in (("seed_0", seed_0), ("seed_1", seed_1)):
        legacy.validate_bitstring(seed, label)
    same_composition = (len(seed_0) == len(seed_1)
                        and seed_0.count("1") == seed_1.count("1"))
    distinct_classes = legacy.canonical_rotation(seed_0) != legacy.canonical_rotation(seed_1)
    # legacy 関数は同じ巡回同値類の入力では早期 False となるため、
    # self-swap の判定には上記の明示的な列挙が必要。
    same_length = len(seed_0) == len(seed_1)
    cross_01 = same_length and legacy.obtainable_by_one_adjacent_swap(seed_0, seed_1, periodic=True)
    cross_10 = same_length and legacy.obtainable_by_one_adjacent_swap(seed_1, seed_0, periodic=True)
    self_0 = has_self_equivalent_adjacent_swap(seed_0)
    self_1 = has_self_equivalent_adjacent_swap(seed_1)
    user_condition = same_composition and distinct_classes and not cross_01 and not cross_10
    return {
        "same_composition": bool(same_composition),
        "cyclic_inequivalent": bool(distinct_classes),
        "cross_swap_01": bool(cross_01),
        "cross_swap_10": bool(cross_10),
        "self_swap_0": bool(self_0),
        "self_swap_1": bool(self_1),
        "adjacent_swap_valid": bool(user_condition),
        "paper_seed_condition": bool(user_condition and not self_0 and not self_1),
    }


def theory_table(prepared):
    """prepare が生成した windows/models から理論比較表を作る。

    predicted_window_correctable=True は十分条件による保証。
    保証の外側は False（訂正不能）ではなく pd.NA（未判定）。
    """
    import pandas as pd

    rows = []
    condition_cache = {}
    for window in prepared["windows"].to_dict("records"):
        pair_id, n = window["pair_id"], int(window["inflation_n"])
        model = prepared["models"][(pair_id, n)]
        seeds = (model.seed_0, model.seed_1)
        if seeds not in condition_cache:
            condition_cache[seeds] = paper_seed_conditions(*seeds)
        conditions = condition_cache[seeds]
        N, W = int(window["N"]), int(window["window_size"])
        predicted_N = seeds[0].count("0") * fibonacci(n) + seeds[0].count("1") * fibonacci(n + 1)
        if N != model.N or N != predicted_N:
            raise ValueError("inflation の長さが原論文の N=k0*f_n+k1*f_(n+1) と一致しません。")
        length = L_theory(n)
        applicable = n >= 1 and conditions["paper_seed_condition"]
        guaranteed = applicable and 0 <= W <= min(N, length)
        rows.append({
            **window, "n": n, "W": W, "seed_0": seeds[0], "seed_1": seeds[1],
            "L_theory": length, "W_le_L_theory": W <= length,
            **conditions, "theory_guarantee_applicable": bool(applicable),
            "predicted_window_correctable": True if guaranteed else pd.NA,
            "theory_window_guaranteed": bool(guaranteed),
            "theory_reference": PAPER_REFERENCE,
        })
    columns = ["pair_id", "inflation_n", "n", "N", "window_size", "W", "seed_0", "seed_1",
               "L_theory", "W_le_L_theory", "same_composition", "cyclic_inequivalent",
               "cross_swap_01", "cross_swap_10", "self_swap_0", "self_swap_1",
               "adjacent_swap_valid", "paper_seed_condition", "theory_guarantee_applicable",
               "predicted_window_correctable", "theory_window_guaranteed", "theory_reference"]
    result = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    result["predicted_window_correctable"] = result["predicted_window_correctable"].astype("boolean")
    return result
