"""Model registry lifecycle states and allowed transitions (MLOps-style)."""

from __future__ import annotations

from typing import Final

# Typical flow: train → evaluate → benchmark → production → retire
LifecycleState = str

STATE_CREATED: Final = "created"
STATE_TRAINING: Final = "training"
STATE_TRAINED: Final = "trained"
STATE_TESTING: Final = "testing"
STATE_BENCHMARKED: Final = "benchmarked"
STATE_PRODUCTION: Final = "production"
STATE_RETIRED: Final = "retired"

ALL_STATES: frozenset[str] = frozenset(
    {
        STATE_CREATED,
        STATE_TRAINING,
        STATE_TRAINED,
        STATE_TESTING,
        STATE_BENCHMARKED,
        STATE_PRODUCTION,
        STATE_RETIRED,
    }
)

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_CREATED: frozenset({STATE_TRAINING, STATE_RETIRED}),
    STATE_TRAINING: frozenset({STATE_TRAINED, STATE_RETIRED}),
    STATE_TRAINED: frozenset({STATE_TESTING, STATE_BENCHMARKED, STATE_PRODUCTION, STATE_RETIRED}),
    STATE_TESTING: frozenset({STATE_BENCHMARKED, STATE_PRODUCTION, STATE_TRAINED, STATE_RETIRED}),
    STATE_BENCHMARKED: frozenset({STATE_PRODUCTION, STATE_TESTING, STATE_RETIRED}),
    STATE_PRODUCTION: frozenset({STATE_RETIRED, STATE_TESTING}),
    STATE_RETIRED: frozenset({STATE_TRAINED}),  # un-archive to trained for re-eval
}

SERVING_STATES: frozenset[str] = frozenset(
    {STATE_TRAINED, STATE_TESTING, STATE_BENCHMARKED, STATE_PRODUCTION}
)


def assert_transition(current: str, new: str) -> None:
    if new not in ALL_STATES:
        raise ValueError(f"Unknown lifecycle state: {new}")
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if new not in allowed:
        raise ValueError(
            f"Cannot transition from '{current}' to '{new}'. "
            f"Allowed: {', '.join(sorted(allowed)) or 'none'}"
        )


def state_guidance(state: str) -> str:
    tips = {
        STATE_CREATED: "Record exists; start a training job to produce artifacts.",
        STATE_TRAINING: "Training in progress; registry updates when the job succeeds.",
        STATE_TRAINED: "Artifacts on disk; run manual or automated tests before production.",
        STATE_TESTING: "In evaluation—use /v1 for smoke tests; not yet production-approved.",
        STATE_BENCHMARKED: "Benchmark metrics captured; review model card before promotion.",
        STATE_PRODUCTION: "Approved for inference via /v1/chat/completions.",
        STATE_RETIRED: "Archived; not served. Artifacts retained for audit until deleted.",
    }
    return tips.get(state, "")
