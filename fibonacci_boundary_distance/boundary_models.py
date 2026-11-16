"""Explicit candidate sets for classical open chains and the unchanged ring code.

Classical recoverability is the prior Notebook's *within-class* injectivity:
    x[not E] == y[not E]  =>  x == y, for x,y in C_i, separately for i=0,1.
It does not require distinguishing the two logical classes.  The only open
candidate sets implied by fixed seed words and a fixed origin are the two
singletons.  Their recoverability is identically True, including for E=all sites;
this is a degenerate reference, not evidence about physical boundary effects.

A caller can explicitly supply a classical candidate factory.  No cyclic
orbit, translation, reflection, padding, or infinite-language completion is
silently added to that open candidate set.  Quantum evaluation is disabled
unless the caller explicitly declares the one state family supported here:
positive equal-amplitude superpositions over distinct supplied candidates.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import hashlib

import recoverability_kl as prior


inflate = prior.inflate
validate_pair = prior.validate_pair
normalize_erasure = prior.normalize_erasure

SOURCE_DIRECTORY = (
    "/Users/ideryuito/Documents/Codex/2026-10-06/"
    "files-pasted-by-the-user-fibonacci/outputs/fibonacci_recoverability_kl"
)
SOURCE_PROVENANCE = {
    "recoverability_kl.py": SOURCE_DIRECTORY + "/recoverability_kl.py",
    "legacy_functions.py": SOURCE_DIRECTORY + "/legacy_functions.py",
    "seed_pair_reference": (
        "/Users/ideryuito/Documents/Codex/2026-10-06/"
        "files-pasted-by-the-user-fibonacci-3/outputs/"
        "fibonacci_finite_size_scaling/reference/seed_pairs.jsonl"
    ),
    "prior_seed_pair": {"pair_id": "L06_W03_P0001", "seed_0": "000111", "seed_1": "010101"},
    "definition_lines": {
        "inflation": "recoverability_kl.py:25-26",
        "seed_validation": "recoverability_kl.py:41-55",
        "classical_recoverability": "recoverability_kl.py:101-112,166-178",
        "KL_integer_counts": "recoverability_kl.py:124-178",
        "ring_logical_states": "recoverability_kl.py:195-214",
    },
}

UNIFORM_STATE_FORMULA = (
    "|Psi_i> = |C_i|^(-1/2) sum_{x in C_i} |x>, i=0,1; "
    "C_i contains distinct words and all amplitudes are positive real"
)
SINGLETON_DESCRIPTION = (
    "Fixed-origin open words: C_0={inflate(seed_0,n)}, "
    "C_1={inflate(seed_1,n)}. Within-class recoverability is trivially True "
    "for every erasure and cannot diagnose boundary sensitivity."
)


def source_provenance():
    """Return portable hashes of the bundled, unchanged prior implementations."""
    info = dict(SOURCE_PROVENANCE)
    root = Path(__file__).resolve().parent
    info["bundled_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("recoverability_kl.py", "legacy_functions.py")
    }
    return info


def _candidate_set(candidates, N, name):
    if isinstance(candidates, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of distinct binary words, not a word")
    try:
        words = tuple(candidates)
    except TypeError as exc:
        raise ValueError(f"{name} must be iterable") from exc
    if not words:
        raise ValueError(f"{name} must be nonempty")
    for word in words:
        prior.legacy.validate_bitstring(word, name)
        if len(word) != N:
            raise ValueError(f"Every {name} word must have the inflation-derived length N={N}")
    if len(set(words)) != len(words):
        raise ValueError(f"{name} contains duplicate words; supply distinct candidates explicitly")
    return tuple(sorted(words))


def _signature(words):
    return hashlib.sha256("\n".join(words).encode("ascii")).hexdigest()


def _classical_evaluate(words0, words1, erased_mask, full_mask):
    """Only within-class collisions are relevant for prior recoverability."""
    remaining_mask = full_mask ^ erased_mask
    counts0 = Counter(word & remaining_mask for word in words0)
    counts1 = Counter(word & remaining_mask for word in words1)
    collision0 = sum(size * (size - 1) // 2 for size in counts0.values())
    collision1 = sum(size * (size - 1) // 2 for size in counts1.values())
    cross = sum(counts0[key] * counts1[key] for key in counts0.keys() & counts1.keys())
    return dict(
        recoverable=collision0 == collision1 == 0,
        recoverable_both=collision0 == collision1 == 0,
        recoverable_0=collision0 == 0,
        recoverable_1=collision1 == 0,
        self_collision_count_0=collision0,
        self_collision_count_1=collision1,
        cross_collision_count=cross,
        KL_diagonal=None,
        KL_offdiagonal=None,
        KL_correctable=None,
    )


class RingModel:
    """The prior finite-ring cyclic-orbit code, with normalized field names."""

    def __init__(self, seed0, seed1, n):
        self._model = prior.LogicalPair(seed0, seed1, n)
        self.N, self.F0, self.F1 = self._model.N, self._model.F0, self._model.F1
        self.C0, self.C1 = tuple(self._model.O0), tuple(self._model.O1)
        self.seed_0, self.seed_1, self.inflation_n = seed0, seed1, n
        self.support_count_0, self.support_count_1 = len(self.C0), len(self.C1)
        self.metadata = dict(
            self._model.metadata,
            geometry="ring",
            model_id="cyclic_orbit_ring",
            model_description="Original finite-ring code over distinct full cyclic orbits.",
            recoverability_definition="within_class_injectivity_both",
            support_count_0=self.support_count_0,
            support_count_1=self.support_count_1,
            support_size_0=self.support_count_0,
            support_size_1=self.support_count_1,
            support_sha256_0=_signature(sorted(self.C0)),
            support_sha256_1=_signature(sorted(self.C1)),
            open_kl_enabled=False,
            quantum_state_definition=UNIFORM_STATE_FORMULA,
            degenerate_singleton_support=False,
            open_support_is_singleton=False,
        )

    def evaluate(self, E):
        result = self._model.evaluate(E)
        result["recoverable"] = result["recoverable_both"]
        return result


class OpenModel:
    """A fixed-coordinate, explicitly supplied classical candidate-set model."""

    def __init__(self, seed0, seed1, n, candidate_factory=None,
                 model_id="anchored_singleton", model_description=None,
                 quantum_state_definition=None, enable_kl=False):
        seed_metadata = validate_pair(seed0, seed1)
        self.inflation_n = prior.require_int(n, "inflation_n")
        self.seed_0, self.seed_1 = seed0, seed1
        self.F0, self.F1 = inflate(seed0, n), inflate(seed1, n)
        if len(self.F0) != len(self.F1):
            raise ValueError("Inflated open words must have equal length")
        self.N = len(self.F0)
        if not isinstance(enable_kl, bool):
            raise ValueError("enable_kl must be True or False")
        self.enable_kl = enable_kl
        if candidate_factory is None:
            raw0, raw1 = (self.F0,), (self.F1,)
            if model_id != "anchored_singleton":
                raise ValueError("A custom model_id requires an explicit candidate_factory")
            model_description = model_description or SINGLETON_DESCRIPTION
        else:
            if not callable(candidate_factory):
                raise ValueError("candidate_factory must be callable")
            raw_pair = candidate_factory(self.F0, self.F1, self.inflation_n)
            try:
                raw0, raw1 = raw_pair
            except (TypeError, ValueError) as exc:
                raise ValueError("candidate_factory must return exactly (C0, C1)") from exc
            if model_id == "anchored_singleton":
                model_id = "user_defined_candidates"
            if not isinstance(model_description, str) or not model_description.strip():
                raise ValueError("An explicit model_description is required for a custom candidate_factory")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        self.C0 = _candidate_set(raw0, self.N, "C0")
        self.C1 = _candidate_set(raw1, self.N, "C1")
        self.words0, self.words1 = tuple(int(x, 2) for x in self.C0), tuple(int(x, 2) for x in self.C1)
        self.support_count_0, self.support_count_1 = len(self.C0), len(self.C1)
        self._full_mask = (1 << self.N) - 1
        self.model_id, self.model_description = model_id, model_description
        state_definition = None
        if self.enable_kl:
            if (not isinstance(quantum_state_definition, Mapping)
                    or quantum_state_definition.get("family") != "equal_positive_uniform_support"
                    or not isinstance(quantum_state_definition.get("description"), str)
                    or not quantum_state_definition["description"].strip()):
                raise ValueError(
                    "Open KL requires quantum_state_definition={"
                    "'family':'equal_positive_uniform_support', 'description':'explicit model rationale'}"
                )
            if set(self.C0) & set(self.C1):
                raise ValueError("Open KL requires disjoint supports for orthogonal logical states")
            state_definition = dict(quantum_state_definition, formula=UNIFORM_STATE_FORMULA)
        elif quantum_state_definition is not None:
            raise ValueError("Set enable_kl=True to supply a quantum_state_definition; otherwise leave it None")
        self.metadata = dict(
            seed_metadata,
            geometry="open",
            seed_0=seed0,
            seed_1=seed1,
            inflation_n=self.inflation_n,
            N=self.N,
            model_id=self.model_id,
            model_description=self.model_description,
            recoverability_definition="within_class_injectivity_both",
            support_count_0=self.support_count_0,
            support_count_1=self.support_count_1,
            support_size_0=self.support_count_0,
            support_size_1=self.support_count_1,
            support_sha256_0=_signature(self.C0),
            support_sha256_1=_signature(self.C1),
            open_kl_enabled=self.enable_kl,
            quantum_state_definition=state_definition,
            degenerate_singleton_support=self.support_count_0 == self.support_count_1 == 1,
            open_support_is_singleton=self.support_count_0 == self.support_count_1 == 1,
            periodic_adjacency=False,
            candidate_origin="fixed_origin_inflated_words" if candidate_factory is None else "explicit_user_factory",
        )

    def evaluate(self, E):
        E = normalize_erasure(E, self.N)
        erased_mask = prior.legacy.erased_site_mask(E, self.N)
        if not self.enable_kl:
            result = _classical_evaluate(self.words0, self.words1, erased_mask, self._full_mask)
        else:
            groups0 = prior.group_integer_words(self.words0, erased_mask, self._full_mask)
            groups1 = prior.group_integer_words(self.words1, erased_mask, self._full_mask)
            result = prior.classify_groups(groups0, groups1, self.support_count_0, self.support_count_1)
            result["recoverable"] = result["recoverable_both"]
        return dict(result, erasure_size=len(E), erasure_positions=E)


def build_ring_model(seed0, seed1, n):
    return RingModel(seed0, seed1, n)


def build_open_model(seed0, seed1, n, candidate_factory=None,
                     model_id="anchored_singleton", model_description=None,
                     quantum_state_definition=None, enable_kl=False):
    return OpenModel(seed0, seed1, n, candidate_factory, model_id, model_description,
                     quantum_state_definition, enable_kl)
