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
