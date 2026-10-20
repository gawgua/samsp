# MaxSAT performance goal: Set I

Work started on 2026-10-11 at the user's request. The goal is to make MaxSAT faster than CP on every one of the 27 current Set I JSON files, including upper-bound construction in total runtime. This document records work from that request forward. Earlier benchmark artifacts are preserved.

**Final observed result:** the optimized backend is faster on all 27 Set I files in each of three fresh paired repetitions: 81 MaxSAT wins, zero CP wins. All 162 schedules independently validate, and every pair proves the same optimum. Total wall time is 221.447 seconds for CP and 111.426 seconds for MaxSAT. The smallest median ratio is 1.032; the closest cases still have limited margin against future timing variation. Full results are linked below.

## Acceptance conditions

The final comparison must use fresh sequential runs, alternating backend order, identical input files and Python environment, a 120-second total wall budget, and a 4096 MB combined process RSS guard. CP uses the existing model with one worker. MaxSAT cannot consume CP incumbents, final optima, cached schedules, or instance-specific answers. All returned schedules require independent validation against input lags, weighted renewable demands, and both aggregate storage bounds. An improvement counts only when both backends prove the same optimum. Every Set I instance must be faster; aggregate speedup alone does not satisfy the goal. Repeat measurements will distinguish substantial gains from startup and sampling noise.

## Initial evidence and investigation

- The existing complete comparison has CP solving all 27 instances, with MaxSAT resolving 13 and stopping for memory on 14. Its large computed horizon is a major construction bottleneck.
- Compact order variables and predefined CP bounds have already reduced construction, but the latter require CP and cannot establish standalone superiority.
- The arithmetic upper-bound heuristic is independent of CP, SAT and MaxSAT. Its previous measured preparation cost reaches approximately 1.55 seconds on some mixed instances, already exceeding the historical CP total on those instances.
- Checked a suspected pseudo-Boolean auxiliary-variable collision. PySAT WCNF.append updates its variable count from appended clauses, so this hypothesis is rejected; no fix is necessary.
- Prototyped a unary-resource subset lower bound using earliest starts and completion tails. Nonempty subsets give a bound equal to the historical optimum on 11 files. Historical optima are used only to assess the diagnostic, never as input to the bound calculation. Empty subsets must be excluded; an initial scratch diagnostic omitted this condition and was corrected before implementation.

## Planned first experiments

1. Implement and exhaustively test the general resource subset lower bound.
2. Replace expensive signed storage rows with capacity constraints over certified storage intervals where input timing proves every paired release follows acquisition. Use the aggregate encoding as a fallback when certification fails.
3. Avoid timing variables for unused renewable activities, simplify constant order literals, and compare supported cardinality encodings and RC2 engines.
4. Measure independent upper-bound construction and strengthen its scheduling choices if its cost or bound blocks progress.

The following experiments record the changes and their measured outcomes in order.

## Experiment 1: compact intervals and resource bounds

Implemented a separate optimized MaxSAT backend, retaining the historical SAT and MaxSAT solvers. It simplifies constants, allocates renewable usage only when needed, supports weighted renewable demands, and uses certified unit storage intervals. Uncertified or weighted storage falls back to signed aggregate constraints enforcing both bounds. Infeasible trial horizons expand; their failure is not reported as global infeasibility.

The first complete in-process pilot, with five-second optimization limits, returned 21 validated optima and six unresolved results. It is diagnostic, excludes interpreter startup, and is not a fresh CP comparison. Raw results and schedules are saved under `benchmark_results/maxsat_goal/pilot_01`. Several small instances complete in approximately 0.03–0.3 seconds including loading and construction. The largest pure type-2 instance still costs approximately 3.14 seconds. Mixed-instance proofs remain the main bottleneck.

Added resource-saturation implications and residue domains when equal durations must exactly fill a certified workload interval. Pilot validation caught an incorrect decoder boundary for a skipped earliest start. Fixed the window to its first and last admissible values and added a dedicated regression. Four focused tests pass, including 54 exhaustive tiny configurations, weighted resources, signed storage fallback, empty objectives, and a saturated resource domain.

Tried Glucose4, MiniSat22, MiniSat-GH, MapleChrono, MapleCM, and Maplesat on three hard mixed instances with three-second optimization limits. None resolved those cases under that limit. CaDiCaL is available but its local PySAT wrapper does not support interruption; timed experiments require the external process guard. No engine superiority claim follows from these short trials.

Additional storage-energy subset bounds did not improve the renewable lower bound on any Set I file. Resource-energy propagation between arbitrary anchor activities leaves the global bounds unchanged but strengthens many internal windows: 5220 pairwise lags on `{220}.json`, with 125 stronger earliest starts and 86 stronger completion tails. Testing these stronger windows and forced disjunctive orders is the next experiment.

## Experiment 2: propagation, grouping, and idle-slot constraints

Added resource-implied disjunctive orders, energy propagation between anchor activities, and conditional window propagation under each trial horizon. Grouped renewable activities and storage stays into temporally certified non-overlap chains before counting capacity. Replaced repeated hard-clause variable scans with direct WCNF hard-list appends, retaining explicit variable allocation and native cardinality variable counts.

Added a storage rule for chains whose acquisition order and release-completion order are both known. The acquisition at position k must follow release k-capacity. These implications come from the original zone capacity and certified nonnegative intervals; they do not fix an arbitrary storage pairing order. Exhaustive finite validation covers a three-stay, capacity-two example. This raises the computed bounds to the proven optima for several pure-type instances, including `{320}` at 176 and `{420}` at 170.

Added a global idle-slot cardinality constraint for nearly saturated unary-resource subsets. Exact busy literals count the subset's occupation in its release-to-deadline interval. This resolves mixed cases which previously timed out: in-process measurements for `{110-310}`, `{110-410}` and `{310-410}` are approximately 0.7 seconds including loading, all bound calculations, successive formula construction and proof.

Lazy time-point covers and lazy interval-order covers were explored as optional experimental modes. Their short mixed-instance pilots were slower or unresolved. They remain disabled in the measured backend. Tiny exhaustive checks cover eager clauses and both lazy modes. A lazy certificate-posting defect found during a pilot was corrected before verification.

The first fresh complete paired comparison is saved in `benchmark_results/maxsat_goal/paired_20261011-215604`. It includes 54 sequential cold worker runs with alternating backend order, a 120-second total wall budget, a 4096 MB combined RSS guard, and one CP worker. Both backends prove all 27 matching optima and every returned schedule passes validation. MaxSAT wins 18 instances and CP wins nine. This does not satisfy the goal. The next experiment removes unused NetworkX construction and historical storage constants from the optimized backend's input pipeline, while preserving the CP baseline.

An attempted historical SAT regression import is unavailable because the current workspace lacks `solver/samsp_sat_cpm2_solver.py`. This file was already absent when the goal began; it is not recreated or silently substituted in the benchmark. The current CP and historical MaxSAT files are preserved.

## Experiment 3: input and startup costs

Implemented `dataset/array_problem.py` for the optimized backend. It retains the JSON arrays and computes the temporal closure directly with NumPy, avoiding NetworkX graph construction and historical storage constants that this backend never reads. Compared its closure, initial lower bound, coarse horizon, and all start/completion windows against the existing loader on every Set I input; all 27 match. CP continues using its existing loader and model.

The second fresh paired comparison, `benchmark_results/maxsat_goal/paired_20261011-220044`, proves 27 matching optima per backend with no invalid or unresolved runs. MaxSAT wins 23 files and CP wins four. The four remaining losses are the three `t1x10_t2x10_s*.json` files and `{110-210}.json`; startup and formula construction contribute substantially to these short runs.

Import profiling found that PySAT eagerly loads the optional AIG conversion package although this solver constructs WCNF clauses directly. Added a lazy package proxy that loads the actual converter on first attribute access. This preserves AIG conversion rather than disabling it. A fresh-process check constructs a CNF from an AIG expression successfully after importing the optimized solver.

Ran three complete repetitions with the same alternating sequential protocol. Saved in `benchmark_results/maxsat_goal/paired_20261011-220647`: 162 validated optimal runs, all paired optima identical, MaxSAT faster on all 27 instance medians. It wins 79 of 81 individual pairs. Two third-repeat pairs favor CP by approximately 2 and 4 milliseconds. Minimum median ratio is 1.036, with total CP wall 218.950 seconds and MaxSAT wall 96.632 seconds. The independent checker imports no solver, problem class, or existing schedule validator; it validates every schedule directly against raw JSON and checks input, frozen source and CP binary hashes.

## Experiment 4: directional capacity predicates

Profiling `{110-210}.json` shows more than 112,000 calls to the hard-clause builder. Changed conjunction predicates used only by capacity upper bounds to the implication `(a AND b) implies v`. An actual occupation must set v true, so an overload cannot be hidden. Conversely, every valid schedule extends to a satisfying assignment by setting v exactly to its conjunction. Predicates used in resource-saturation or idle-slot lower bounds retain full equivalence. Cached predicates can be upgraded to full equivalence when needed.

This reduces `{110-210}.json` from 115,249 to 88,889 hard clauses without changing its optimum or objective. The 54 exhaustive configurations, weighted-resource fallback, saturated-domain regression, FIFO-storage implication test, and both experimental lazy modes still pass. Clause reduction can weaken propagation: preliminary repeats show `{210-310}.json` taking roughly 7 seconds instead of 3, while remaining faster than CP at roughly 14 seconds. This tradeoff is recorded rather than described as a universal improvement.

Added a standalone artifact verifier and median report generator, `verify_maxsat_goal_comparison.py`. Detailed CSV output includes runtime ranges, encoding and optimization times, native solving time, peak RSS, variable counts, hard and soft clause counts, computed lower bound, final horizon and objective cost. Unresolved runs would receive a 120-second penalty; no unresolved run occurs in the completed three-repeat campaign above.

Checked general-runner integration separately: optimized storage metadata differs from historical CPM2 storage keys. Corrected configuration/resume keys so the optimized backend runs once per input and completed optimized runs are skipped correctly. Saved optimized engine settings must agree on resume. Source manifests now hash the actual solver and dataset modules rather than failing on pre-existing absent historical files. These edits affect only the general CLI's manifest and resume handling; worker execution and the benchmark driver's protocol are unchanged.

## Final verified comparison

The final campaign is [paired_20261011-221422](benchmark_results/maxsat_goal/paired_20261011-221422/median_comparison.md). It contains 162 fresh sequential cold worker runs: 27 inputs, two backends, three repetitions. Input order is activity count then filename, with backend order alternating across instances and repetitions. The same Python executable and environment run both backends. CP uses the unchanged CP model and one CP Optimizer worker. MaxSAT uses RC2 with Glucose4, incremental mode, sequential-counter cardinality, zero initial slack, grouping enabled, and adaptation and lazy separation disabled. The total wall limit is 120 seconds, the combined worker/descendant RSS guard is 4096 MB, and the common guard polling interval is 0.02 seconds.

- MaxSAT wins all 81 individual pairs and all 27 instance medians. CP wins zero; there are no ties, unresolved runs, invalid schedules, or mismatched optima.
- All 162 schedules pass the separate raw-JSON event-sweep validator. It checks nonnegative integer starts, source time, every input lag, completion before the sink, weighted renewable occupancy, both storage bounds, simultaneous event semantics, saved schedule consistency, objective cost, and source/input/CP binary hashes.
- Total measured wall is 221.447 seconds for CP and 111.426 seconds for MaxSAT, a ratio of 1.987. Penalized totals are the same because no run requires the 120-second unresolved penalty.
- Median ratios range from 1.032 for `t1x10_t2x10_s1.json` to 3.935 for `{25-45}.json`. Peak observed combined RSS is approximately 111.63 MB for MaxSAT and 104.06 MB for CP.
- The final directional encoding is retained because every measured pair wins. Its aggregate time is worse than the preceding full-equivalence candidate on some hard cases; that earlier candidate and its two millisecond-scale losses remain preserved and reported. No per-instance backend or historical optimum selects the configuration.

The [median CSV](benchmark_results/maxsat_goal/paired_20261011-221422/median_comparison.csv) includes phase times, native times, runtime ranges, peak RSS, final formula variable and hard/soft clause counts, objective cost, computed lower bound, and final horizon. [Raw paired results](benchmark_results/maxsat_goal/paired_20261011-221422/results.jsonl), [manifest](benchmark_results/maxsat_goal/paired_20261011-221422/manifest.json), [independent verification](benchmark_results/maxsat_goal/paired_20261011-221422/verification.json), and [resume verification](benchmark_results/maxsat_goal/paired_20261011-221422/resume_verification.json) are retained alongside backend logs, schedules and frozen sources.

## Current input-to-solution flow

1. [SAMSPProblem](dataset/problems.py) loads the input and computes max-plus timing closure. The array parser has been consolidated into this shared parser. The coarse legacy horizon is retained as an input-derived field, not accepted as a feasible incumbent.
2. [Resource bound functions](solver/resource_bounds.py) infer forced non-overlap orders, strengthen timing distances using required resource work, and compute nonempty-subset release/energy/tail lower bounds.
3. [Storage certification and FIFO propagation](solver/samsp_maxsat_optimized_solver.py) match unit acquisitions to releases only when timing proves each interval nonnegative. Known acquisition and release orders then imply capacity-driven lag constraints. Uncertified storage uses the signed aggregate encoding with both bounds.
4. The trial horizon starts at the computed lower bound in the measured configuration. This is a feasibility question, not a certified upper bound. No CP call, saved CP bound, saved makespan, heuristic schedule, or cached solution is used.
5. `encode()` propagates windows under that trial horizon and allocates compact order literals. `get_L(i,t)` returns false below the earliest start and true at or beyond the latest start; shared constants and residue domains avoid physical variables for every time from zero to the coarse horizon.
6. The builder appends hard timing, renewable and storage clauses directly into WCNF. Certified non-overlap groups reduce each capacity row. Directional conjunctions suffice for capacity upper bounds; exact conjunctions and exact busy disjunctions support saturation and idle-slot lower bounds.
7. The builder enforces the lower bound and appends weight-one soft sink units from the lower bound through the trial horizon minus one. Other activity literals never enter the objective. Unsatisfied soft weight equals sink makespan minus the lower bound.
8. `solve()` runs RC2. An infeasible bounded trial expands the horizon geometrically and rebuilds the formula. The deadline includes all subsequent construction. A feasible completed RC2 result proves the global optimum because every better makespan also fits within this horizon. Its decoded cost must agree before OPTIMAL is reported. Timers are canceled and joined, and native resources are released. Interrupted work returns UNKNOWN.
9. The worker decodes and validates the schedule. The parent records total cold wall time, including imports, loading, preparation, construction, proof, validation and cleanup. The independent artifact verifier checks the schedules again without importing the implementation.

On files where the lower bound is achievable at the initial horizon, there are no soft clauses. RC2 still proves hard feasibility at a certified lower bound, which certifies optimality. Other files use a nonempty soft objective after horizon expansion. This is expected behavior, not a saved answer supplied to the optimizer.

## Verification and repository preservation

Sixteen focused test methods pass: seven optimized solver methods, three pipeline methods, and six existing arithmetic upper-bound methods. The exhaustive test includes 54 tiny configurations in each of three encoding modes. Additional checks cover weighted resources, uncertified signed storage, nonzero bounds, an empty/fixed objective, infeasible timing, interrupted optimization, saturated-domain decoding, FIFO storage implications, all-27 loader parity, deferred AIG conversion, both runner backends, external timeout reporting, distinct artifacts, general CLI resume, and rejection of changed optimized settings on resume.

Resuming the final full campaign launches zero workers and leaves all 325 result/log/schedule artifacts unchanged. The live runner differs from its frozen copy only in general `main()` manifest/resume handling; an AST comparison verifies that all worker functions and execution code match. The measured optimized solver, bound functions, array loader and other production modules still match their frozen hashes. Extra regression tests were added after freezing and run against current code.

Python compilation and diff whitespace checks pass for the edited production files. The repository-wide whitespace check reports pre-existing CRLF/trailing-whitespace edits in `main.py`; that unrelated file is preserved. The historical SAT regression remains unavailable because the historical SAT solver file was already absent. No datasets, old results, papers, CP model, historical MaxSAT solver, unrelated edits, dependency versions, or installed engines were changed.

## Reproduction

Run from the repository source directory with the same Python environment:

```sh
rtk proxy /home/nguen/pysat/bin/python /mnt/c/users/khoin/downloads/sat/samsp/src/run_maxsat_goal_comparison.py --slack 0 --repeats 3
rtk proxy /home/nguen/pysat/bin/python /mnt/c/users/khoin/downloads/sat/samsp/src/verify_maxsat_goal_comparison.py /mnt/c/users/khoin/downloads/sat/samsp/src/benchmark_results/maxsat_goal/paired_20261011-221422
```

The general runner backend is `maxsat_optimized`. Specify `--maxsat-slack 0 --maxsat-engine g4 --maxsat-cardinality seqcounter` to reproduce the measured solver configuration. The comparison driver applies these settings and the shared sequential protocol by default. Results establish the requested performance goal for the current Set I files, machine, versions and one-worker CP baseline; they do not guarantee the same ordering on another environment or every future run.

## Post-benchmark cleanup: 2026-10-12

At the user's request, removed the unused experimental lazy time-point and interval-order modes. Their short pilots were slower or unresolved, and neither measured benchmark enables them. Deleted `_lazy_cuts` and its nested `precedes`, `cover` and `interval_cover` helpers, unused combinations/islice imports, lazy-only caches and counters, the separation loop, and the `lazy`/`lazy_ordering` constructor options. Removed `lazy_rounds` and `learned_capacity_cuts` statistics, which described only those modes. The active encoder, bound functions, grouping, weighted-storage fallback, objective, horizon expansion and timer/resource handling are retained.

Updated the exhaustive test to exercise the supported eager mode rather than removed experiments. The same ten solver/pipeline tests pass before and after cleanup, including all 54 tiny arithmetic configurations, signed storage, interruption, infeasibility, FIFO propagation, loader parity and runner/resume behavior. Python compilation and scoped whitespace checks pass.

Compared the cleaned encoder against the frozen pre-cleanup solver from the Set II campaign. All 89 comparisons match exactly in variable IDs/counts, hard-clause order/content, soft clauses/weights, lower bounds, time windows and order/usage literals. Coverage includes every Set I file at its initial and distinct saved final horizons, the 54 tiny configurations, and weighted signed-storage fallback. Evidence is saved in [verification_20261012.json](benchmark_results/maxsat_cleanup/verification_20261012.json). Historical benchmark sources and results are preserved. This is a behavior-preserving cleanup; no fresh runtime improvement is claimed.
