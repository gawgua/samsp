"""Certified resource-energy lower bounds, independent of any solver result."""

import numpy as np


def disjunctive_resource_closure(problem, closure, horizon=None):
    """Promote forced orders between jobs that cannot overlap on a resource.

    An order j-before-i is impossible if its completion lag closes a positive
    cycle with the existing i-to-j lag. Non-overlap therefore forces i-before-j.
    A finite horizon adds a conditional sink-to-source lag before propagation.
    """
    closure = closure.copy()
    if horizon is not None:
        a, b = problem.sink_id, problem.source_id
        if closure[b, a] > horizon:
            return None
        np.maximum(closure, closure[:, a, None] - horizon + closure[None, b, :], out=closure)
    for _ in range(problem.number_of_activities):
        changed = False
        for resource, capacity in enumerate(problem.renewable_capacities):
            jobs = [i for i in problem.activity_ids if problem.durations[i] > 0
                    and problem.renewable_requests[i][resource] > 0]
            if not jobs:
                continue
            durations = np.asarray([problem.durations[i] for i in jobs])
            demands = np.asarray([problem.renewable_requests[i][resource] for i in jobs])
            distances = closure[np.ix_(jobs, jobs)]
            forced = ((demands[:, None] + demands[None, :]) > capacity) & (distances + durations[None, :] > 0)
            np.fill_diagonal(forced, False)
            values = np.where(forced, np.maximum(distances, durations[:, None]), distances)
            if not np.array_equal(values, distances):
                closure[np.ix_(jobs, jobs)] = values
                changed = True
        if not changed:
            return closure
        for k in problem.activity_ids:
            np.maximum(closure, closure[:, k, None] + closure[None, k, :], out=closure)
        if np.any(np.diag(closure) > 0):
            return None
    return closure


def strengthen_resource_lags(problem, closure, rounds=2):
    """Add implied lags from resource energy between arbitrary anchor events.

    Only jobs provably starting after anchor a and finishing before anchor b
    enter a subset. Its minimum head, energy/capacity and minimum tail bound
    the distance from a to b. All-pairs propagation preserves implication.
    """
    closure = closure.copy()
    for _ in range(rounds):
        tightened = closure.copy()
        for resource, capacity in enumerate(problem.renewable_capacities):
            jobs = [i for i in problem.activity_ids if problem.durations[i] > 0
                    and problem.renewable_requests[i][resource] > 0]
            if capacity <= 0 or not jobs:
                continue
            heads = closure[:, jobs]
            tails = closure[jobs, :] - np.asarray([problem.durations[i] for i in jobs])[:, None]
            common = (heads[:, :, None] >= 0) & (tails[None, :, :] >= 0)
            energies = np.asarray([problem.durations[i] * problem.renewable_requests[i][resource] for i in jobs])
            totals = ((heads >= 0).astype(np.int64) * energies) @ (tails >= 0).astype(np.int64)
            minimum_head = np.min(np.where(common, heads[:, :, None], np.inf), axis=1)
            minimum_tail = np.min(np.where(common, tails[None, :, :], np.inf), axis=1)
            candidate = minimum_head + minimum_tail + (totals + capacity - 1) // capacity
            candidate[totals == 0] = -np.inf
            np.maximum(tightened, candidate, out=tightened)
        if np.array_equal(closure, tightened):
            break
        closure = tightened
        for k in problem.activity_ids:
            np.maximum(closure, closure[:, k, None] + closure[None, k, :], out=closure)
        if np.any(np.diag(closure) > 0):
            return None
    return closure


def resource_lower_bound(problem, closure):
    """Bound nonempty subsets with common release and completion-tail bounds.

    If all jobs in a subset start at or after r, and leave at least q time
    before the sink after completion, their total resource energy fits into
    capacity * (makespan - r - q). Nonnegative starts and completion-before-
    sink must be included in closure. Empty subsets contribute no bound.
    """
    bound = int(closure[problem.source_id, problem.sink_id])
    witnesses = []
    for resource, capacity in enumerate(problem.renewable_capacities):
        if capacity <= 0:
            continue
        jobs = [i for i in problem.activity_ids
                if problem.durations[i] > 0 and problem.renewable_requests[i][resource] > 0]
        if not jobs:
            continue
        releases = np.asarray([int(closure[problem.source_id, i]) for i in jobs])
        tails = np.asarray([int(closure[i, problem.sink_id]) - problem.durations[i] for i in jobs])
        energy = np.asarray([problem.durations[i] * problem.renewable_requests[i][resource] for i in jobs])
        rs, qs = np.unique(releases), np.unique(tails)
        cells = np.zeros((len(rs), len(qs)), dtype=np.int64)
        np.add.at(cells, (np.searchsorted(rs, releases), np.searchsorted(qs, tails)), energy)
        totals = cells[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
        values = rs[:, None] + qs[None, :] + (totals + capacity - 1) // capacity
        values[totals == 0] = -1
        row, column = np.unravel_index(np.argmax(values), values.shape)
        candidate = int(values[row, column])
        if candidate > bound:
            bound = candidate
        witnesses.append(dict(resource=resource, release=int(rs[row]), tail=int(qs[column]),
                              energy=int(totals[row, column]), capacity=capacity, lower_bound=candidate))
    return bound, witnesses
