# Arithmetic upper-bound heuristic

`find_upper_bound(problem)` in `schedule_upper_bound.py` finds a feasible schedule without invoking CP, SAT or MaxSAT. It returns the first independently validated schedule and its sink start as a certified upper bound. It is a feasibility heuristic and does not prove optimality or infeasibility.

The heuristic starts with the max-plus closure of the supplied lag graph. It adds nonnegative-start and completion-before-sink arcs to a private copy. Earliest start times from this closure satisfy the temporal constraints. When a renewable resource is overloaded, the heuristic tries ordering one overlapping activity after another's completion. When storage exceeds capacity, it tries completing a release before an acquisition. When storage becomes negative, it tries an acquisition before a release's completion. All storage events at the same timestamp are aggregated together.

Each added arc updates the closure with an outer-product relaxation. An arc that would create a positive cycle is rejected. Candidate orders prioritize storage releases, then estimated makespan and total start-time delay. Bounded backtracking tries alternative recent orders, and deterministic randomized restarts diversify the search. Each closure update takes quadratic time in the activity count. The heuristic does not pair storage items or change the solver's hard formula.

The default heuristic budget is two seconds, with eight attempts and seed zero. Time is checked cooperatively between repair operations. Recent backtracking history is limited to 64 states, and each attempt allows at most four times the activity count in repair iterations. These limits can cause UNKNOWN on a feasible instance. An UNKNOWN result contains neither a schedule nor an upper bound.

Before returning a bound, the production schedule validator checks every supplied lag, completion before the sink, weighted renewable demand, and both aggregate storage bounds. The problem graph, horizon and windows remain unchanged during the heuristic.

Use the returned bound before constructing MaxSAT:

```python
from solver.schedule_upper_bound import find_upper_bound
from solver.samsp_maxsat_cpm2_solver import SAMSPMaxSAT_CPM2_Solver

result = find_upper_bound(problem, time_limit=2.0)
if result.upper_bound is not None:
    problem.set_search_horizon(result.upper_bound)
    solver = SAMSPMaxSAT_CPM2_Solver(problem)
    solver.encode()
```

For an end-to-end runtime comparison, include problem loading, preprocessing, the heuristic, validation, horizon recomputation, MaxSAT construction and optimization in one wall budget.

`run_schedule_upper_bounds.py` measures bound construction on all 27 Set I files. It reads historical CP first-incumbent bounds only for reporting and input-hash verification; those values do not enter the heuristic. The recorded run in `benchmark_results/arithmetic_ub/20261010-114701` produced 27 independently validated schedules. Preparation, heuristic and independent validation totaled 10.770 seconds, excluding shared imports and interpreter startup. Bounds were tighter than the historical CP initial bound on seven files, equal on four, and looser on sixteen. No MaxSAT speedup has been measured with these new bounds.

Six focused heuristic tests and sixteen existing CPM2 tests passed. A separate raw JSON verifier rechecked all 27 benchmark schedules. A tiny handoff check confirmed that applying the returned bound before allocation creates a MaxSAT formula with the corresponding horizon and preserves the optimum.

The temporal conflict-repair idea is related to Cesta, Oddi and Smith's RCPSP/max approach: https://publications.ri.cmu.edu/a-constraint-based-method-for-project-scheduling-with-time-windows. The storage repair rules and bounded implementation here are repository-specific and do not claim to reproduce that paper's full algorithm.
