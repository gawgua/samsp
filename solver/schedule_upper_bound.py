"""Find a certified scheduling upper bound without CP or a Boolean encoding.

Resolve resource conflicts by adding temporal ordering arcs to a private
max-plus closure. This is a greedy feasibility heuristic, not an exact solver.
Failure or exhaustion of its budget provides no infeasibility proof.
"""

from dataclasses import dataclass
from itertools import combinations
import random
import time

import numpy as np


@dataclass
class UpperBoundResult:
    status: str
    upper_bound: int | None
    schedule: dict[int, int] | None
    wall_seconds: float
    attempts: int
    repairs: int


def _add_arc(closure, source, target, weight):
    """Strengthen a closed lag graph in place, rejecting positive cycles."""
    if weight <= closure[source, target]:
        return False
    if closure[target, source] + weight > 0:
        return False
    paths = closure[:, source, None] + weight + closure[None, target, :]
    np.maximum(closure, paths, out=closure)
    return True


def _closure(problem):
    n = problem.number_of_activities
    result = np.full((n, n), -np.inf)
    np.fill_diagonal(result, 0)
    for i, j, data in problem.extended_precedence_graph.edges(data=True):
        result[i, j] = data["weight"]
    # Include nonnegative starts and completion before the sink in the heuristic.
    # Do not modify the problem's graph, windows, or SAT/MaxSAT constraints.
    for i in problem.activity_ids:
        for a, b, weight in ((problem.source_id, i, 0),
                             (i, problem.sink_id, problem.durations[i])):
            if weight > result[a, b] and not _add_arc(result, a, b, weight):
                return None
    if np.any(np.diag(result) > 0):
        return None
    return result


def _conflict(problem, starts):
    """Find the earliest overload or underflow with simultaneous event semantics."""
    first = None
    for r, capacity in enumerate(problem.renewable_capacities):
        events = {}
        for i in problem.activity_ids:
            demand = problem.renewable_requests[i][r]
            if not demand or not problem.durations[i]:
                continue
            events.setdefault(starts[i], []).append((i, demand))
            events.setdefault(starts[i] + problem.durations[i], []).append((i, -demand))
        use, active = 0, set()
        for t in sorted(events):
            for i, change in events[t]:
                use += change
                if change > 0:
                    active.add(i)
                else:
                    active.discard(i)
            if use > capacity:
                candidate = (t, "renewable", r, tuple(sorted(active)))
                if first is None or t < first[0]:
                    first = candidate
                break
    for r, capacity in enumerate(problem.cumulative_capacities):
        events = {}
        for i in problem.activity_ids:
            delta = problem.cumulative_requests[i][r]
            if delta:
                t = starts[i] + (problem.durations[i] if delta < 0 else 0)
                events[t] = events.get(t, 0) + delta
        use = 0
        for t in sorted(events):
            use += events[t]
            if not 0 <= use <= capacity:
                candidate = (t, "storage_upper" if use > capacity else "storage_lower", r, ())
                if first is None or t < first[0]:
                    first = candidate
                break
    return first


def _arcs(problem, starts, conflict):
    t, kind, r, active = conflict
    if kind == "renewable":
        for i, j in combinations(active, 2):
            yield i, j, problem.durations[i]
            yield j, i, problem.durations[j]
        return
    gets = [i for i in problem.activity_ids if problem.cumulative_requests[i][r] > 0]
    releases = [i for i in problem.activity_ids if problem.cumulative_requests[i][r] < 0]
    if kind == "storage_upper":
        gets = sorted((i for i in gets if starts[i] <= t), key=lambda i: (-starts[i], i))
        releases = sorted((i for i in releases if starts[i] + problem.durations[i] > t),
                          key=lambda i: (starts[i] + problem.durations[i], i))
        for release in releases:
            for get in gets:
                yield release, get, problem.durations[release]
    else:
        gets = sorted((i for i in gets if starts[i] > t), key=lambda i: (starts[i], i))
        releases = sorted((i for i in releases if starts[i] + problem.durations[i] <= t),
                          key=lambda i: (-starts[i] - problem.durations[i], i))
        for get in gets:
            for release in releases:
                yield get, release, -problem.durations[release]


def find_upper_bound(problem, *, time_limit=2.0, restarts=8, seed=0):
    """Return the first fully validated feasible schedule, or UNKNOWN.

    Time includes closure setup and validation, but not creation of ``problem``.
    The budget is checked cooperatively between bounded repair operations.
    ``upper_bound`` is populated only for a validated schedule. Neither the
    supplied problem nor its windows are changed. No historical bounds are read.
    """
    started = time.monotonic()
    deadline = started + max(0, time_limit)
    attempts = repairs = 0

    def finish(schedule=None):
        return UpperBoundResult("FEASIBLE" if schedule is not None else "UNKNOWN",
                                schedule[problem.sink_id] if schedule is not None else None,
                                schedule, time.monotonic() - started, attempts, repairs)

    if time_limit <= 0 or restarts <= 0:
        return finish()
    base = _closure(problem)
    if base is None:
        return finish()
    rng = random.Random(seed)
    durations = np.asarray(problem.durations)
    for attempt in range(restarts):
        if time.monotonic() >= deadline:
            break
        attempts += 1
        closure = base.copy()
        history = []
        # Bound each attempt so an unlucky order cannot consume every restart.
        for _ in range(4 * problem.number_of_activities):
            if time.monotonic() >= deadline:
                return finish()
            starts = closure[problem.source_id].astype(int)
            conflict = _conflict(problem, starts)
            if conflict is None:
                # Reuse the independently maintained complete schedule validator.
                from run_dataset import validate
                schedule = {i: int(starts[i]) for i in problem.activity_ids}
                if not validate(problem, schedule):
                    return finish(schedule)
                break
            candidates = []
            for a, b, weight in _arcs(problem, starts, conflict):
                if time.monotonic() >= deadline:
                    return finish()
                if weight <= closure[a, b] or closure[b, a] + weight > 0:
                    continue
                delayed = np.maximum(starts, starts[a] + weight + closure[b])
                storage_priority = sum(problem.cumulative_requests[a])
                score = (storage_priority, int(np.max(delayed + durations)), int(np.sum(delayed - starts)),
                         -closure[a, problem.sink_id], a, b)
                candidates.append((score, (a, b, weight)))
                if len(candidates) == 16:
                    break
            if not candidates:
                # Undo recent heuristic orders rather than declaring infeasibility.
                while history:
                    previous, alternatives = history.pop()
                    if alternatives:
                        arc = alternatives.pop(0)
                        closure = previous.copy()
                        history.append((previous, alternatives))
                        _add_arc(closure, *arc)
                        repairs += 1
                        break
                else:
                    break
                continue
            candidates.sort(key=lambda candidate: candidate[0])
            choice = 0 if attempt == 0 else rng.randrange(min(3, len(candidates)))
            alternatives = [arc for index, (_, arc) in enumerate(candidates) if index != choice]
            history.append((closure.copy(), alternatives))
            if len(history) > 64:
                del history[0]
            _add_arc(closure, *candidates[choice][1])
            repairs += 1
    return finish()
