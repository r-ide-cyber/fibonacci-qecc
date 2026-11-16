"""既存Notebookから関数本体をそのまま抽出。出典・SHA256はreference/provenance.json。"""

from __future__ import annotations

from itertools import combinations

import numpy as np

def validate_bitstring(bitstring: str, name: str) -> None:
    """非空の二進文字列を検証する。"""
    if not isinstance(bitstring, str) or not bitstring:
        raise ValueError(f"{name}は空でない文字列にしてください。")
    if set(bitstring) - {"0", "1"}:
        raise ValueError(f"{name}には0と1だけを使用してください。")

def cyclic_orbit(bitstring: str) -> list[str]:
    """重複を除いた巡回シフト軌道を返す。"""
    validate_bitstring(bitstring, "bitstring")
    return list(
        dict.fromkeys(
            bitstring[shift:] + bitstring[:shift]
            for shift in range(len(bitstring))
        )
    )

def canonical_rotation(bitstring: str) -> str:
    """巡回シフトの辞書順最小代表を返す。"""
    return min(cyclic_orbit(bitstring))

def inflate(
    seed: str,
    depth: int,
    substitution: dict[str, str],
) -> str:
    """置換規則をdepth回適用する。"""
    validate_bitstring(seed, "seed")
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise ValueError("depthは整数にしてください。")
    if depth < 0:
        raise ValueError("depthは0以上にしてください。")
    result = seed
    for _ in range(depth):
        result = "".join(substitution[bit] for bit in result)
    return result

def obtainable_by_one_adjacent_swap(
    source: str,
    target: str,
    periodic: bool = True,
) -> bool:
    """1回の異なる隣接ビット交換で巡回同値になるか判定する。"""
    validate_bitstring(source, "source")
    validate_bitstring(target, "target")
    if len(source) != len(target):
        raise ValueError("比較するビット列の長さが違います。")
    if canonical_rotation(source) == canonical_rotation(target):
        return False

    target_canonical = canonical_rotation(target)
    last_left = len(source) if periodic else len(source) - 1
    for left in range(last_left):
        right = (left + 1) % len(source)
        if source[left] == source[right]:
            continue
        swapped = list(source)
        swapped[left], swapped[right] = (
            swapped[right],
            swapped[left],
        )
        if canonical_rotation("".join(swapped)) == target_canonical:
            return True
    return False

def fixed_weight_representatives(
    length: int,
    number_of_ones: int,
) -> list[str]:
    """固定重み二進列の巡回同値類代表を返す。"""
    representatives: set[str] = set()
    for one_sites in combinations(range(length), number_of_ones):
        bits = ["0"] * length
        for site in one_sites:
            bits[site] = "1"
        representatives.add(canonical_rotation("".join(bits)))
    return sorted(representatives)

def deserialize_integer_state(
    serialized: dict[str, list[float]],
) -> IntegerState:
    """二進基底文字列を整数ビット状態へ変換する。"""
    return [
        (int(bits, 2), complex(parts[0], parts[1]))
        for bits, parts in serialized.items()
    ]

def erased_site_mask(
    sites: tuple[int, ...],
    n: int,
) -> int:
    """左から数えた消失サイト集合を整数マスクへ変換する。"""
    mask = 0
    for site in sites:
        if site < 0 or site >= n:
            raise ValueError("消失サイトが範囲外です。")
        mask |= 1 << (n - 1 - site)
    if mask.bit_count() != len(sites):
        raise ValueError("消失サイトが重複しています。")
    return mask

def group_integer_state(
    state: IntegerState,
    erased_mask: int,
    full_mask: int,
) -> GroupedState:
    """補領域の整数ごとに消失部分と振幅をまとめる。"""
    from collections import defaultdict

    remaining_mask = full_mask ^ erased_mask
    grouped: dict[int, dict[int, complex]] = defaultdict(dict)
    for basis, amplitude in state:
        grouped[basis & remaining_mask][basis & erased_mask] = amplitude
    return dict(grouped)

def cross_operator(
    grouped_a: GroupedState,
    grouped_b: GroupedState,
) -> SparseOperator:
    """共通補領域を持つ項だけから疎な交差演算子を作る。"""
    from collections import defaultdict

    operator: dict[tuple[int, int], complex] = defaultdict(complex)
    for remaining in grouped_a.keys() & grouped_b.keys():
        for row, amplitude_a in grouped_a[remaining].items():
            for column, amplitude_b in grouped_b[remaining].items():
                operator[(row, column)] += (
                    amplitude_a * amplitude_b.conjugate()
                )
    return {
        key: value
        for key, value in operator.items()
        if value != 0.0
    }

def frobenius_norm(operator: SparseOperator) -> float:
    """疎演算子のFrobeniusノルムを返す。"""
    import math

    return math.sqrt(
        sum(abs(value) ** 2 for value in operator.values())
    )

def difference_norm(
    operator_a: SparseOperator,
    operator_b: SparseOperator,
    scale: float,
) -> float:
    """scale*(A-B)のFrobeniusノルムを疎に返す。"""
    import math

    return math.sqrt(
        sum(
            abs(
                scale
                * (
                    operator_a.get(key, 0.0)
                    - operator_b.get(key, 0.0)
                )
            ) ** 2
            for key in operator_a.keys() | operator_b.keys()
        )
    )

def sparse_kl_residual(
    states: list[IntegerState],
    sites: tuple[int, ...],
    n: int,
    tolerance: float,
) -> dict[str, object]:
    """密行列を作らず1消失配置のKL残差を計算する。"""
    full_mask = (1 << n) - 1
    erased_mask = erased_site_mask(sites, n)
    groups = [
        group_integer_state(state, erased_mask, full_mask)
        for state in states
    ]

    operator_00 = cross_operator(groups[0], groups[0])
    operator_11 = cross_operator(groups[1], groups[1])
    operator_01 = cross_operator(groups[0], groups[1])

    diagonal = difference_norm(
        operator_00,
        operator_11,
        scale=0.5,
    )
    off_diagonal = frobenius_norm(operator_01)
    residual = max(diagonal, off_diagonal)
    return {
        "diagonal_residual": diagonal,
        "off_diagonal_residual": off_diagonal,
        "max_residual": residual,
        "satisfies_kl": residual <= tolerance,
    }
