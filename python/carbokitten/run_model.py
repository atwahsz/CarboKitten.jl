from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar


class HasInitialState(Protocol):
    def initial_state(self):  # pragma: no cover
        ...


S = TypeVar("S")


class Model(Protocol[S]):
    """A runnable model protocol.

    A model provides:
    - initial_state() -> S
    - step(state: S) -> tuple[S, frame]

    The returned `frame` can be any object (or None).
    """

    steps: int

    def initial_state(self) -> S:  # pragma: no cover
        ...

    def step(self, state: S):  # pragma: no cover
        ...


def run_model(model: Model[S], *, callback: Callable[[int, object], None] | None = None) -> S:
    """Run a model for its configured number of steps.

    Parameters
    - model: a model instance (already configured with its input)
    - callback: called as callback(step_index_1_based, frame)

    Returns
    - The final state.
    """

    state = model.initial_state()
    for w in range(1, model.steps + 1):
        state, frame = model.step(state)
        if callback is not None:
            callback(w, frame)
    return state


def run_model_with_writers(
    model: Model[S],
    *,
    state_writer: Callable[[int, S], None] | None = None,
    frame_writer: Callable[[int, object], None] | None = None,
) -> S:
    """Run a model and write both state and per-step frames.

    This mirrors the Julia pattern where output writers receive both the evolving
    state and the per-step `Frame` values.
    """

    state = model.initial_state()
    if state_writer is not None:
        state_writer(1, state)

    for w in range(1, model.steps + 1):
        state, frame = model.step(state)
        if frame_writer is not None:
            frame_writer(w, frame)
        if state_writer is not None:
            state_writer(w + 1, state)
    return state
