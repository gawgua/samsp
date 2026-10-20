"""Run dataset JSONs with a wall-clock limit and inspectable timeout reports.

Example: python run_dataset.py --time-limit 120 --jobs 2
The limit includes imports, preprocessing, encoding, and solving. Results and
logs are written under benchmark_results/.
Workers and descendants have a combined 4096 MB RSS guard by default.
--memory-limit-mb 0 disables it. Limit stops report UNKNOWN for CPM2.
Reported wall time includes cleanup after the termination request.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import faulthandler
import importlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import hashlib
import platform

ROOT = Path(__file__).resolve().parent
BACKENDS = {
    "sat_cpm2": ("solver.samsp_sat_cpm2_solver", "SAMSPSAT_CPM2_Solver"),
    "sat_cpm2_res_uni": ("solver.samsp_sat_cpm2_solver", "SAMSPSAT_CPM2_Solver"),
    "maxsat_cpm2": ("solver.samsp_maxsat_cpm2_solver", "SAMSPMaxSAT_CPM2_Solver"),
    "maxsat_cpm2_res_uni": ("solver.samsp_maxsat_cpm2_solver", "SAMSPMaxSAT_CPM2_Solver"),
    "cp": ("solver.samsp_cp_solver", "SAMSPCPSolver"),
    "milp": ("solver.samsp_milp_solver", "SAMSPMILPSolver"),
    "gurobi": ("solver.samsp_gurobi_solver", "SAMSPGurobiSolver"),
}
STUDIO_CP = Path("/opt/ibm/ILOG/CPLEX_Studio222/cpoptimizer/bin/x86-64_linux/cpoptimizer")
CPM2_BACKENDS = {"sat_cpm2", "sat_cpm2_res_uni", "maxsat_cpm2", "maxsat_cpm2_res_uni"}
MAXSAT_BACKENDS = {"maxsat_cpm2", "maxsat_cpm2_res_uni"}
SAT_BACKENDS = {"sat", *CPM2_BACKENDS}


def sat_wrapper(solver):
    return getattr(solver, f"_{type(solver).__name__}__solver", None)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite(value):
    return value if value is not None and math.isfinite(value) else None


def validate(problem, schedule):
    """Check supplied lags, makespan, and simultaneous resource events."""
    violations = []
    if schedule[problem.source_id] != 0:
        violations.append("source does not start at zero")
    for i in problem.activity_ids:
        if schedule[i] < 0:
            violations.append(f"activity {i} starts before zero")
        completion = schedule[i] + problem.durations[i]
        if completion > schedule[problem.sink_id]:
            violations.append(f"activity {i} finishes at {completion}, after sink {schedule[problem.sink_id]}")
    for arc in problem.lags:
        i, j, lag = arc["from"], arc["to"], arc["lag"]
        if schedule[j] - schedule[i] < lag:
            violations.append(f"lag {i}, {j} violated")
    for kind, requests, capacities in (
        ("renewable", problem.renewable_requests, problem.renewable_capacities),
        ("storage", problem.cumulative_requests, problem.cumulative_capacities),
    ):
        for r, capacity in enumerate(capacities):
            events = {}
            for i in problem.activity_ids:
                demand, start, duration = requests[i][r], schedule[i], problem.durations[i]
                if not demand:
                    continue
                if kind == "renewable":
                    if not duration:
                        continue
                    changes = ((start, demand), (start + duration, -demand))
                else:
                    changes = ((start + (duration if demand < 0 else 0), demand),)
                for t, change in changes:
                    events[t] = events.get(t, 0) + change
            occupancy = 0
            for t in sorted(events):
                occupancy += events[t]
                if not 0 <= occupancy <= capacity:
                    violations.append(f"{kind} {r} at time {t}: {occupancy}, capacity {capacity}")
    return violations


def worker(args):
    started = time.monotonic()
    deadline = started + args.time_limit
    state = {"solver": args.worker, "instance": str(args.instance),
             "stage": "imports", "status": "RUNNING", "model": {}}
    if args.worker in CPM2_BACKENDS:
        state.update(storage_semantics="upper_only_historical",
                     optimization="maxsat" if args.worker in MAXSAT_BACKENDS else "sat")
    objects = {}
    main_thread = threading.get_ident()
    stop = threading.Event()
    snapshot_lock = threading.RLock()
    faulthandler.enable()

    def record_schedule(problem, schedule):
        violations = validate(problem, schedule)
        state.update(schedule=schedule, schedule_found=True,
                     makespan=schedule[problem.sink_id], valid=not violations,
                     violations=violations[:20], violation_count=len(violations))

    def make_snapshot():
        frames = []
        sat_search = None
        frame = sys._current_frames().get(main_thread)
        while frame is not None:
            frames.append(frame)
            obj = frame.f_locals.get("self")
            if obj is not None:
                name = type(obj).__name__
                if name == "SAMSPProblem":
                    objects.setdefault("problem", obj)
                elif name in {item[1] for item in BACKENDS.values()}:
                    objects.setdefault("solver", obj)
                    if (name in {"TEMP", "SAMSPSAT_CPM2_Solver", "SAMSPMaxSAT_CPM2_Solver"}
                            and frame.f_code.co_name in {"solve", "__solve_sat"} and sat_search is None):
                        sat_search = {key: frame.f_locals.get(value) for key, value in
                                      (("tested_makespan", "makespan"), ("lower_bound", "low"),
                                       ("upper_bound", "high"), ("last_feasible_bound", "best"))}
            frame = frame.f_back
        project_frames = [f for f in frames if f.f_code.co_filename.startswith(str(ROOT))
                          and f.f_code.co_filename != __file__]
        location = (project_frames or frames)[0] if frames else None
        report = dict(state)
        if sat_search is not None:
            report["search"] = sat_search
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["peak_memory_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        if state.get("timeout_location"):
            report["location"] = state["timeout_location"]
        elif location:
            report["location"] = {"file": location.f_code.co_filename,
                                  "function": location.f_code.co_name,
                                  "line": location.f_lineno}
        if frames:
            report["active_frame"] = {"file": frames[0].f_code.co_filename,
                                      "function": frames[0].f_code.co_name,
                                      "line": frames[0].f_lineno}
            if state["status"] not in {"TIMEOUT", "UNKNOWN"}:
                args.snapshot.with_suffix(".stack.txt").write_text("\n".join(
                    f'{f.f_code.co_filename}:{f.f_lineno} in {f.f_code.co_name}' for f in frames
                ) + "\n")
        metrics = dict(state.get("model", {}))
        problem = objects.get("problem")
        if problem:
            report["activities"] = getattr(problem, "number_of_activities", None)
            report["LB"] = getattr(problem, "LB", None)
            report["UB"] = getattr(problem, "UB", None)
        solver = objects.get("solver")
        if solver is not None:
            if args.worker in SAT_BACKENDS:
                metrics.update(solver.get_statistics())
                model = solver.get_last_feasible_model()
                metrics.update(incumbent_available=model is not None,
                               solve_calls=metrics["number_of_calls"])
                if model is not None and model is not objects.get("recorded_model"):
                    starts = solver.L if args.worker in CPM2_BACKENDS else solver.S
                    schedule = {i: next(t for t in range(problem.ES[i], problem.LS[i] + 1)
                                        if model[starts[i, t] - 1] > 0)
                                for i in problem.activity_ids}
                    record_schedule(problem, schedule)
                    objects["recorded_model"] = model
                    report.update({key: state[key] for key in
                                   ("schedule", "schedule_found", "makespan", "valid",
                                    "violations", "violation_count")})
                for name in ("S", "x", "y", "L", "u", "register"):
                    metrics[name + "_variables"] = len(getattr(solver, name, {}))
            elif args.worker == "milp" and hasattr(solver, "model"):
                metrics.update(variables=solver.model.number_of_variables,
                               constraints=solver.model.number_of_constraints)
            elif args.worker == "cp" and hasattr(solver, "model"):
                metrics.update(variables=len(solver.tasks) + len(solver.occupancy),
                               expressions=len(solver.model.get_all_expressions()))
            elif args.worker == "gurobi":
                metrics["start_variables"] = len(getattr(solver, "x", {}))
                metrics.setdefault("variables", metrics["start_variables"] if hasattr(solver, "x") else None)
                metrics.setdefault("constraints", None)
                if problem is not None and hasattr(problem, "LC"):
                    metrics["planned_start_variables"] = int(sum(
                        max(0, lc - es + 1) for es, lc in zip(problem.ES, problem.LC)
                    ))
        report["model"] = metrics
        write_json(args.snapshot, report)

    def snapshot():
        with snapshot_lock:
            make_snapshot()

    def monitor():
        while not stop.wait(1):
            try:
                snapshot()
            except Exception as error:
                print(f"Snapshot error: {error}", file=sys.stderr, flush=True)

    snapshot()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        from dataset.problems import SAMSPProblem
        module, name = BACKENDS[args.worker]
        solver_class = getattr(importlib.import_module(module), name)
        state["stage"] = "preprocess"
        problem = SAMSPProblem.from_json(args.instance)
        objects["problem"] = problem
        if args.worker in MAXSAT_BACKENDS and getattr(args, "maxsat_heuristic_ub", False):
            from dataclasses import asdict
            from solver.schedule_upper_bound import find_upper_bound
            state["stage"] = "heuristic_ub"
            state["original_UB"] = problem.UB
            bound = find_upper_bound(problem, time_limit=min(
                args.heuristic_time_limit, max(0, deadline - time.monotonic())))
            state["heuristic_ub"] = asdict(bound)
            if bound.schedule is not None:
                violations = validate(problem, bound.schedule)
                if violations:
                    raise ValueError(f"Invalid heuristic upper bound: {violations[:3]}")
                problem.set_search_horizon(bound.upper_bound)
            state["heuristic_ub"]["applied"] = bound.schedule is not None
            state["heuristic_ub"]["search_horizon"] = problem.UB
            snapshot()
        upper_bound = getattr(args, "cpm2_upper_bound", None)
        if upper_bound is None and args.worker in MAXSAT_BACKENDS:
            upper_bound = getattr(args, "maxsat_upper_bound", None)
        if args.worker in CPM2_BACKENDS and upper_bound is not None:
            state["original_UB"] = problem.UB
            problem.set_search_horizon(upper_bound)
            state["predefined_UB"] = upper_bound
        state["stage"] = "initialize"
        if args.worker in CPM2_BACKENDS:
            encoding = ("res_uni" if args.worker.endswith("_res_uni")
                        else getattr(args, "renewable_encoding", "amo"))
            state["renewable_encoding"] = encoding
            solver = solver_class(problem, renewable_encoding=encoding)
        else:
            solver = solver_class(problem)
        objects["solver"] = solver
        state["stage"] = "encode" if args.worker in SAT_BACKENDS else "build_model"
        build_started = time.monotonic()
        if args.worker in SAT_BACKENDS:
            if args.worker in CPM2_BACKENDS:
                solver.encode(deadline=deadline)
            else:
                solver.encode()
        else:
            solver.build_model()
        state["build_seconds"] = time.monotonic() - build_started
        remaining = max(0.001, deadline - time.monotonic())
        if args.worker == "gurobi":
            from gurobipy import GRB
            solver.model.update()
            state["model"] = {"variables": solver.model.NumVars,
                              "constraints": solver.model.NumConstrs,
                              "binary_variables": solver.model.NumBinVars,
                              "integer_variables": solver.model.NumIntVars}
            solver.model.Params.TimeLimit = remaining

            def progress(model, where):
                if where == GRB.Callback.MIP:
                    incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
                    bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                    state["search"] = {"incumbent": incumbent if abs(incumbent) < GRB.INFINITY else None,
                                       "best_bound": bound if abs(bound) < GRB.INFINITY else None,
                                       "nodes": model.cbGet(GRB.Callback.MIP_NODCNT)}
        elif args.worker == "milp":
            from docplex.mp.progress import ProgressListener

            class Progress(ProgressListener):
                def notify_progress(self, data):
                    state["search"] = {"incumbent": finite(data.current_objective) if data.has_incumbent else None,
                                       "best_bound": finite(data.best_bound),
                                       "gap": finite(data.mip_gap), "nodes": data.current_nb_nodes}

            solver.model.add_progress_listener(Progress())
            solver.model.set_time_limit(remaining)
        elif args.worker == "cp":
            from docplex.cp.solver.solver import CpoCallback
            statistics = solver.model.get_statistics()
            state["model"] = {"declared_variables": statistics.get_number_of_variables(),
                              "constraints": statistics.nb_constraints,
                              "interval_variables": statistics.nb_interval_vars,
                              "integer_variables": statistics.nb_integer_vars,
                              "sequence_variables": statistics.nb_sequence_vars}

            class Progress(CpoCallback):
                def invoke(self, engine, event, result):
                    search = dict(state.get("search", {}))
                    search.update(event=event, solver_status=str(result.get_solve_status()))
                    for key, value in (("objective", result.get_objective_values()),
                                       ("bound", result.get_objective_bounds())):
                        if value is not None:
                            search[key] = value
                    state["search"] = search
                    if result.is_solution():
                        task_solutions = {i: result.get_var_solution(task)
                                          for i, task in solver.tasks.items()}
                        if all(solution is not None for solution in task_solutions.values()):
                            record_schedule(problem, {
                                i: solution.get_start() for i, solution in task_solutions.items()
                            })

            solver.model.add_solver_callback(Progress())
        elif args.worker not in CPM2_BACKENDS:
            wrapper = sat_wrapper(solver)
            original_solve = wrapper.solve

            def limited_solve(time_limit=None):
                answer = original_solve(time_limit=max(0.001, deadline - time.monotonic()))
                snapshot()
                return answer

            wrapper.solve = limited_solve
        state["stage"] = "solve"
        optimization_started = time.monotonic()
        if args.worker == "cp":
            options = {"TimeLimit": remaining}
            if args.cp_workers is not None:
                options["Workers"] = args.cp_workers
            if args.cp_executable:
                options["execfile"] = str(args.cp_executable)
            state["cp_executable"] = str(args.cp_executable or "cpoptimizer")
            solver.solution = solver.model.solve(**options)
        elif args.worker == "gurobi":
            solver.model.optimize(progress)
        elif args.worker in CPM2_BACKENDS:
            solve_status = solver.solve(time_limit=max(0.001, deadline - time.monotonic()))
        else:
            solve_status = solver.solve()
        state["optimization_seconds"] = time.monotonic() - optimization_started
        solve_snapshot = json.loads(args.snapshot.read_text())
        if args.worker in SAT_BACKENDS:
            state["search"] = solve_snapshot.get("search", {})
        state["stage"] = "decode"
        schedule = None
        if args.worker in SAT_BACKENDS:
            model = solver.get_last_feasible_model()
            state["solver_status"] = solve_status.name
            if model:
                starts = solver.L if args.worker in CPM2_BACKENDS else solver.S
                schedule = {i: next(t for t in range(problem.ES[i], problem.LS[i] + 1)
                                    if model[starts[i, t] - 1] > 0) for i in problem.activity_ids}
            state["model"] = solver.get_statistics()
            status = ("UNKNOWN" if args.worker in CPM2_BACKENDS else "TIMEOUT") if solve_status.name == "UNKNOWN" else solve_status.name
            if solve_status.name == "UNKNOWN":
                state["stop_reason"] = "time_limit"
        elif args.worker == "cp":
            solution = solver.get_solution()
            state["solver_status"] = str(solution.get_solve_status())
            state["stop_cause"] = str(solution.get_stop_cause())
            state["model"].update(encoding_seconds=state["build_seconds"],
                                  optimization_seconds=state["optimization_seconds"],
                                  total_solving_time=solution.get_solve_time())
            state["objective_bounds"] = solution.get_objective_bounds()
            state["objective_gaps"] = solution.get_objective_gaps()
            if solution.is_solution():
                schedule = {i: solution.get_var_solution(task).get_start() for i, task in solver.tasks.items()}
            status = ("TIMEOUT" if "TimeLimit" in state["stop_cause"]
                      or state["stop_cause"] == "SearchStoppedByLimit"
                      else state["solver_status"].upper())
        elif args.worker == "milp":
            solution = solver.get_solution()
            state["solver_status"] = solver.model.solve_details.status
            state["gap"] = finite(solver.model.solve_details.mip_relative_gap)
            if solution is not None:
                schedule = {i: next(t for t in range(problem.ES[i], problem.LS[i] + 1)
                                    if solution.get_value(solver.x[i, t]) > 0.5) for i in problem.activity_ids}
            native_status = state["solver_status"].lower()
            status = ("TIMEOUT" if "time limit" in native_status else
                      "OPTIMAL" if "optimal" in native_status else
                      "INFEASIBLE" if "infeasible" in native_status else
                      "FEASIBLE" if schedule is not None else "UNKNOWN")
        else:
            state["solver_status"] = solver.model.Status
            schedule = solver.get_scheduling()
            state["gap"] = finite(solver.model.MIPGap) if schedule else None
            status = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIMEOUT",
                      GRB.INFEASIBLE: "INFEASIBLE"}.get(solver.model.Status, f"GUROBI_{solver.model.Status}")
        state["stage"] = "validate"
        state["schedule_found"] = schedule is not None
        if schedule is not None:
            record_schedule(problem, schedule)
            violations = state["violations"]
            if violations and status not in {"TIMEOUT", "UNKNOWN"}:
                status = "INVALID_SOLUTION"
        if status in {"TIMEOUT", "UNKNOWN"}:
            state["timeout_location"] = solve_snapshot.get("location")
            state["timeout_stack"] = str(args.snapshot.with_suffix(".stack.txt"))
        state["status"] = status
        state["stage"] = "complete"
    except TimeoutError as error:
        state.update(status="UNKNOWN", stop_reason="time_limit", error=str(error))
    except Exception as error:
        state.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        stop.set()
        thread.join(timeout=2)
        snapshot()


def process_group_rss_mb(group):
    """Sum RSS for the worker and every process in its Linux process group."""
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) != group:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) / 1024
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def run_one(args, instance, backend, folder):
    relative = instance.relative_to(args.dataset_dir)
    name = "__".join(relative.with_suffix("").parts) + "__" + backend
    snapshot = folder / (name + ".json")
    log = folder / (name + ".log")
    environment = os.environ.copy()
    if args.license:
        environment["GRB_LICENSE_FILE"] = str(args.license)
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
               "--worker", backend, "--instance", str(instance),
               "--snapshot", str(snapshot), "--time-limit", str(args.time_limit)]
    if args.cp_executable:
        command.extend(["--cp-executable", str(args.cp_executable)])
    if args.cp_workers is not None:
        command.extend(["--cp-workers", str(args.cp_workers)])
    upper_bound = getattr(args, "maxsat_upper_bound", None)
    if backend in MAXSAT_BACKENDS and upper_bound is not None:
        command.extend(["--maxsat-upper-bound", str(upper_bound)])
    shared_bound = getattr(args, "cpm2_upper_bound", None)
    if backend in CPM2_BACKENDS and shared_bound is not None:
        command.extend(["--cpm2-upper-bound", str(shared_bound)])
    if backend in CPM2_BACKENDS:
        command.extend(["--renewable-encoding", "res_uni" if backend.endswith("_res_uni")
                        else getattr(args, "renewable_encoding", "amo")])
    if backend in MAXSAT_BACKENDS and getattr(args, "maxsat_heuristic_ub", False):
        command.extend(["--maxsat-heuristic-ub", "--heuristic-time-limit",
                        str(getattr(args, "heuristic_time_limit", 2.0))])
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=stream, env=environment,
                                   start_new_session=True, cwd=ROOT)
        timed_out = False
        memory_stopped = False
        stop_requested = None
        peak_rss = 0
        while process.poll() is None:
            elapsed = time.monotonic() - started
            peak_rss = max(peak_rss, process_group_rss_mb(process.pid))
            timed_out = elapsed >= args.time_limit
            memory_stopped = bool(args.memory_limit_mb and peak_rss >= args.memory_limit_mb)
            if timed_out or memory_stopped:
                stop_requested = round(elapsed, 3)
                break
            time.sleep(getattr(args, "poll_interval", 0.2))
        if timed_out or memory_stopped:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            # A descendant may outlive the worker. Clean up the whole group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    result = json.loads(snapshot.read_text()) if snapshot.exists() else {
        "solver": backend, "instance": str(instance), "stage": "startup", "model": {}}
    if timed_out or memory_stopped:
        result["status"] = "UNKNOWN" if backend in CPM2_BACKENDS else ("MEMORY_LIMIT" if memory_stopped else "TIMEOUT")
        result["stop_reason"] = "memory_limit" if memory_stopped else "time_limit"
        result["timeout_stage"] = "solve" if result.get("stage") == "complete" else result.get("stage")
        result["stop_requested_seconds"] = stop_requested
        result["snapshot_age_seconds"] = round(time.monotonic() - started - result.get("elapsed_seconds", 0), 3)
        stack = snapshot.with_suffix(".stack.txt")
        result["timeout_stack"] = str(stack) if stack.exists() else None
    elif process.returncode != 0:
        result.update(status="ERROR", error=f"Worker exited with code {process.returncode}")
        if process.returncode < 0:
            result["exit_signal"] = signal.Signals(-process.returncode).name
    elif result.get("status") in {"TIMEOUT", "UNKNOWN"}:
        result["timeout_stage"] = "solve" if result.get("stage") == "complete" else result.get("stage")
    result.update(instance=str(instance), solver=backend, exit_code=process.returncode,
                  wall_seconds=round(time.monotonic() - started, 3), time_limit_seconds=args.time_limit,
                  log=str(log))
    if backend in CPM2_BACKENDS:
        result.update(storage_semantics="upper_only_historical",
                      optimization="maxsat" if backend in MAXSAT_BACKENDS else "sat")
    result["memory_limit_mb"] = args.memory_limit_mb
    result["peak_memory_mb"] = max(peak_rss, result.get("peak_memory_mb", 0))
    if stop_requested is not None:
        result["cleanup_seconds"] = round(result["wall_seconds"] - stop_requested, 3)
    write_json(snapshot, result)
    return result


def write_summary(folder, results):
    fields = ["instance", "solver", "status", "makespan", "valid", "wall_seconds",
              "storage_semantics", "stop_reason", "total_solving_time",
              "optimization", "encoding_seconds", "optimization_seconds", "hard_clauses", "soft_clauses", "objective_cost",
              "timeout_stage", "solver_status", "variables", "clauses", "constraints",
              "expressions", "solve_calls", "peak_memory_mb", "location_file",
              "location_line", "location_function", "snapshot_age_seconds", "gap",
              "memory_limit_mb", "stop_requested_seconds", "cleanup_seconds"]
    with (folder / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in sorted(results, key=lambda r: (r["instance"], r["solver"])):
            row = dict(result, **result.get("model", {}))
            for key in ("file", "line", "function"):
                row["location_" + key] = (result.get("location") or {}).get(key)
            writer.writerow(row)
    counts, by_solver, stages = {}, {}, {}
    for result in results:
        status, solver = result["status"], result["solver"]
        counts[status] = counts.get(status, 0) + 1
        group = by_solver.setdefault(solver, {})
        group[status] = group.get(status, 0) + 1
        if status == "TIMEOUT" or result.get("stop_reason") == "time_limit":
            stage = result.get("timeout_stage", result.get("stage"))
            stages[stage] = stages.get(stage, 0) + 1
    summary = {"runs": len(results), "statuses": counts, "by_solver": by_solver,
               "timeout_stages": stages,
               "invalid_schedules": sum(r.get("valid") is False for r in results)}
    write_json(folder / "summary.json", summary)
    return summary




def parse_solver_selection(value):
    """Resolve registered backend names and filename aliases without importing solvers."""
    tokens = [token.strip() for token in value.split(",")]
    if tokens == ["all"]:
        return list(BACKENDS)
    aliases = {name: name for name in BACKENDS}
    for name, (module, _) in BACKENDS.items():
        if name.endswith("_res_uni"):
            continue
        filename = module.rsplit(".", 1)[-1] + ".py"
        aliases[filename] = name
        aliases[filename.removeprefix("samsp_")] = name
    selected = []
    for token in tokens:
        backend = aliases.get(Path(token).name) if token else None
        if backend is None:
            raise ValueError(f"unknown solver {token!r}; choose {', '.join(BACKENDS)}, or all")
        if backend in selected:
            raise ValueError(f"duplicate solver selection: {backend}")
        selected.append(backend)
    return selected


def parse_sets(value):
    selected = tuple(token.strip().upper() for token in value.split(","))
    if not selected or any(token not in {"I", "II"} for token in selected):
        raise argparse.ArgumentTypeError("set must be I, II, or I,II")
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError("set list must not contain duplicates")
    return selected


def select_instances(dataset_dir, sets=None):
    if sets is None:
        return sorted(p for p in dataset_dir.rglob("*.json") if p.name != "test.json")
    files = []
    for label in sets:
        directory = dataset_dir / ("set_" + label.lower())
        selected = sorted(p for p in directory.rglob("*.json") if p.name != "test.json")
        if not selected:
            raise ValueError(f"no dataset JSON files found for Set {label}: {directory}")
        files.extend(selected)
    return files


def comparison_markdown(dataset_dir, files, solvers, results):
    """Compare recorded results without treating an incumbent as an optimality proof."""
    indexed = {(row["instance"], row["solver"]): row for row in results}
    groups = {}
    for path in files:
        relative = path.relative_to(dataset_dir)
        label = {"set_i": "Set I", "set_ii": "Set II"}.get(relative.parts[0], "Instances")
        groups.setdefault(label, []).append(path)
    lines = ["Cells: status / makespan / wall seconds / valid.", ""]

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        lines.append("")

    def cell(row):
        if row is None:
            return "PENDING / — / — / —"
        valid = row.get("valid")
        status = ("INVALID" if valid is False or row["status"] == "INVALID_SOLUTION"
                  else row["status"])
        if row.get("stop_reason") in {"time_limit", "memory_limit"}:
            status += " (" + row["stop_reason"] + ")"
        makespan = row.get("makespan")
        wall = row.get("wall_seconds")
        return " / ".join((status, str(makespan) if makespan is not None else "—",
                           f"{wall:.3f}" if wall is not None else "—",
                           "yes" if valid is True else "NO" if valid is False else "—"))

    def instance_name(path):
        relative = path.relative_to(dataset_dir)
        if relative.parts[0] in {"set_i", "set_ii"}:
            return str(Path(*relative.parts[1:]))
        return str(relative)

    for label, paths in groups.items():
        lines.extend(("## " + label, ""))
        table(["Instance", *solvers],
              [[instance_name(path),
                *(cell(indexed.get((str(path), solver))) for solver in solvers)] for path in paths])
    lines.extend(("## Totals", ""))
    totals = []
    for solver in solvers:
        rows = [row for row in results if row["solver"] == solver]
        invalid = sum(row.get("valid") is False or row["status"] == "INVALID_SOLUTION" for row in rows)
        errors = sum(row["status"] == "ERROR" and row.get("valid") is not False for row in rows)
        optimal = sum(row["status"] == "OPTIMAL" and row.get("valid") is True for row in rows)
        infeasible = sum(row["status"] in {"UNSATISFIABLE", "INFEASIBLE"}
                         and row.get("valid") is not False for row in rows)
        unresolved = len(rows) - invalid - errors - optimal - infeasible
        totals.append([solver, str(optimal), str(infeasible), str(unresolved), str(errors), str(invalid),
                       f"{sum(row.get('wall_seconds') or 0 for row in rows):.3f}"])
    table(["Solver", "Validated optima", "Infeasible", "Unresolved", "Errors", "Invalid", "Wall seconds"], totals)
    return "\n".join(lines)


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    comparison = "--compare" in arguments
    parser = argparse.ArgumentParser(prog="run_solvers.sh" if comparison else None,
                                     description="Run selected solvers on Set I and/or Set II and compare results."
                                     if comparison else __doc__)
    parser.add_argument("--compare", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--solver", required=comparison,
                        help="Comma-separated backend names or solver filenames; all selects every backend")
    parser.add_argument("--set", dest="sets", type=parse_sets,
                        default=("I", "II") if comparison else None,
                        help="I, II, or I,II (shell default: both sets, including all Set I files)")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset/data")
    parser.add_argument("--time-limit", type=float, default=120)
    parser.add_argument("--memory-limit-mb", type=int, default=4096,
                        help="Maximum combined worker/descendant RSS; 0 disables the guard (default: 4096 MB)")
    parser.add_argument("--jobs", type=int, default=1 if comparison else 2)
    parser.add_argument("--cp-workers", type=int, default=1 if comparison else None,
                        help="CP Optimizer threads per run; shell default: 1; otherwise engine configuration")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results" / "solver_comparison"
                        if comparison else ROOT / "benchmark_results")
    parser.add_argument("--resume", type=Path, help="Continue an existing results folder, skipping completed runs")
    parser.add_argument("--retry-overruns", action="store_true",
                        help="With --resume, archive and retry delayed stops, SIGKILL exits, and missing running locations")
    parser.add_argument("--retry-cp-errors", action="store_true",
                        help="With --resume, archive and retry only CP runs with ERROR status")
    parser.add_argument("--retry-solvers", nargs="+", choices=BACKENDS, default=[],
                        help="With --resume, archive and retry every run for the selected solvers")
    parser.add_argument("--cp-executable", type=Path,
                        default=os.environ.get("CPO_EXECFILE", str(STUDIO_CP) if STUDIO_CP.is_file() else None),
                        help="CP Optimizer executable; defaults to installed Linux Studio engine when available")
    parser.add_argument("--license", type=Path, default=os.environ.get("GRB_LICENSE_FILE", "/mnt/c/users/khoin/downloads/gurobi.lic"))
    parser.add_argument("--solvers", nargs="+", choices=BACKENDS)
    parser.add_argument("--renewable-encoding", choices=("amo", "res_uni"), default="amo",
                        help="CPM2 renewable encoding; res-uni also has distinct SAT and MaxSAT backend keys")
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--maxsat-upper-bound", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cpm2-upper-bound", type=int,
                        help="Apply the same predefined horizon to SAT and MaxSAT CPM2")
    parser.add_argument("--maxsat-heuristic-ub", action="store_true",
                        help="Find a CP-free feasible horizon before historical maxsat_cpm2; included in total wall budget")
    parser.add_argument("--heuristic-time-limit", type=float, default=2.0,
                        help="Maximum arithmetic upper-bound search time in seconds (default: 2)")
    parser.add_argument("--poll-interval", type=float, default=0.2,
                        help="Parent wall and RSS guard sampling interval in seconds")
    parser.add_argument("--instance", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    if args.solver is not None:
        if args.solvers is not None:
            parser.error("use either --solver or --solvers")
        try:
            args.solvers = parse_solver_selection(args.solver)
        except ValueError as error:
            parser.error(str(error))
    elif args.solvers is None:
        args.solvers = [name for name in BACKENDS if not name.endswith("_res_uni")]
    if len(set(args.solvers)) != len(args.solvers):
        parser.error("solver list must not contain duplicates")
    if not math.isfinite(args.time_limit) or args.time_limit <= 0 or args.jobs < 1 or args.memory_limit_mb < 0:
        parser.error("time-limit and jobs must be positive; memory limit must be nonnegative")
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("poll-interval must be finite and positive")
    if args.cp_workers is not None and args.cp_workers < 1:
        parser.error("cp-workers must be positive")
    if not math.isfinite(args.heuristic_time_limit) or args.heuristic_time_limit <= 0:
        parser.error("heuristic-time-limit must be finite and positive")
    if args.maxsat_heuristic_ub and args.maxsat_upper_bound is not None:
        parser.error("heuristic and predefined MaxSAT upper bounds are mutually exclusive")
    if args.cpm2_upper_bound is not None and (args.cpm2_upper_bound < 0
            or args.maxsat_upper_bound is not None or args.maxsat_heuristic_ub):
        parser.error("cpm2-upper-bound must be nonnegative and cannot be combined with MaxSAT bound options")
    if args.cp_executable:
        args.cp_executable = args.cp_executable.resolve()
        if not args.cp_executable.is_file() or not os.access(args.cp_executable, os.X_OK):
            parser.error("cp-executable must be an executable file")
    if args.worker:
        worker(args)
        return
    if (args.retry_overruns or args.retry_cp_errors or args.retry_solvers) and not args.resume:
        parser.error("retry options require --resume")
    args.dataset_dir = args.dataset_dir.resolve()
    try:
        files = select_instances(args.dataset_dir, args.sets)
    except ValueError as error:
        parser.error(str(error))
    if not files:
        parser.error("no dataset JSON files found")
    results = []
    if args.resume:
        folder = args.resume.resolve()
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["instances"] != [str(p) for p in files] or manifest["solvers"] != args.solvers:
            parser.error("resume requires the same dataset files and solvers")
        if manifest.get("maxsat_upper_bound") != args.maxsat_upper_bound:
            parser.error("resume requires the same predefined MaxSAT upper bound")
        if manifest.get("cpm2_upper_bound") != args.cpm2_upper_bound:
            parser.error("resume requires the same predefined CPM2 upper bound")
        if manifest.get("renewable_encoding", "amo") != args.renewable_encoding:
            parser.error("resume requires the same renewable encoding")
        if (manifest.get("maxsat_heuristic_ub", False) != args.maxsat_heuristic_ub
                or (args.maxsat_heuristic_ub
                    and manifest.get("heuristic_time_limit") != args.heuristic_time_limit)):
            parser.error("resume requires the same heuristic MaxSAT upper-bound settings")
        for line in (folder / "results.jsonl").read_text().splitlines():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                parser.error("results.jsonl contains an incomplete record")
        if args.retry_overruns or args.retry_cp_errors or args.retry_solvers:
            retry = [r for r in results if
                     (args.retry_overruns and ((r["wall_seconds"] > args.time_limit + 2 and "stop_requested_seconds" not in r)
                     or r.get("exit_code") == -signal.SIGKILL
                     or (r["status"] == "TIMEOUT" and r.get("location", {}).get("function") == "snapshot")))
                     or (args.retry_cp_errors and r["solver"] == "cp" and r["status"] == "ERROR")
                     or r["solver"] in args.retry_solvers]
            if retry:
                archive = folder / "previous_attempts" / time.strftime("%Y%m%d-%H%M%S")
                archive.mkdir(parents=True)
                shutil.copyfile(folder / "results.jsonl", archive / "results_before_retry.jsonl")
                for result in retry:
                    log = Path(result["log"])
                    for source in (log, log.with_suffix(".json"), log.with_suffix(".stack.txt")):
                        if source.exists():
                            shutil.copyfile(source, archive / source.name)
                    previous = dict(result, log=str(archive / log.name))
                    if result.get("timeout_stack"):
                        previous["timeout_stack"] = str(archive / Path(result["timeout_stack"]).name)
                    write_json(archive / log.with_suffix(".json").name, previous)
                results = [r for r in results if r not in retry]
                (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
                print(f"Archived {len(retry)} attempts for retry: {archive}", flush=True)
        manifest.setdefault("resumes", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                   "jobs": args.jobs, "completed": len(results),
                                                   "time_limit_seconds": args.time_limit,
                                                   "memory_limit_mb": args.memory_limit_mb,
                                                   "cp_executable": str(args.cp_executable) if args.cp_executable else None,
                                                   "cp_workers": args.cp_workers,
                                                   "retry_cp_errors": args.retry_cp_errors,
                                                   "retry_solvers": args.retry_solvers})
    else:
        folder = args.output.resolve() / time.strftime("%Y%m%d-%H%M%S")
        folder.mkdir(parents=True, exist_ok=False)
        sources = [Path(__file__), ROOT / "dataset/__init__.py", ROOT / "dataset/problems.py",
                   ROOT / "solver/__init__.py", ROOT / "solver/solver.py"]
        sources.extend(ROOT.joinpath(*BACKENDS[backend][0].split(".")).with_suffix(".py")
                       for backend in args.solvers)
        if args.maxsat_heuristic_ub:
            sources.append(ROOT / "solver/schedule_upper_bound.py")
        if any(backend in CPM2_BACKENDS for backend in args.solvers):
            sources.append(ROOT / "solver/renewable_res_uni.py")
        if args.compare:
            sources.append(ROOT / "run_solvers.sh")
        manifest = {"dataset_dir": str(args.dataset_dir), "instances": [str(p) for p in files],
                    "maxsat_upper_bound": args.maxsat_upper_bound,
                    "renewable_encoding": args.renewable_encoding,
                    "cpm2_upper_bound": args.cpm2_upper_bound,
                    "maxsat_heuristic_ub": args.maxsat_heuristic_ub,
                    "heuristic_time_limit": args.heuristic_time_limit if args.maxsat_heuristic_ub else None,
                    "solvers": args.solvers, "jobs": args.jobs, "time_limit_seconds": args.time_limit,
                    "storage_semantics": {backend: "upper_only_historical"
                                          for backend in args.solvers if backend in CPM2_BACKENDS},
                    "memory_limit_mb": args.memory_limit_mb,
                    "memory_scope": "combined worker and descendants",
                    "cp_executable": str(args.cp_executable) if args.cp_executable else None,
                    "cp_workers": args.cp_workers,
                    "limit_includes": ["imports", "preprocessing", "initialization", "encoding",
                                       "solving", "validation"],
                    "python": sys.version, "platform": platform.platform(),
                    "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sources if p.is_file()}}
    if args.sets is not None:
        manifest.update(sets=args.sets, comparison=args.compare)
    total = len(files) * len(args.solvers)
    write_json(folder / "manifest.json", manifest)
    print(f"{len(files)} instances, {total} runs, limit {args.time_limit:g}s/run, {args.jobs} jobs", flush=True)
    print(f"Results: {folder}", flush=True)
    completed = {(r["instance"], r["solver"]) for r in results}
    print(f"Already completed: {len(results)}; remaining: {total - len(results)}", flush=True)
    with (folder / "results.jsonl").open("a") as output, ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_one, args, path, solver, folder)
                   for index, path in enumerate(files)
                   for solver in (args.solvers[index % len(args.solvers):] + args.solvers[:index % len(args.solvers)]
                                  if args.compare else args.solvers)
                   if (str(path), solver) not in completed]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            output.write(json.dumps(result, allow_nan=False) + "\n")
            output.flush()
            model = result["model"]
            print(f"[{len(results)}/{total}] {Path(result['instance']).name} {result['solver']}: "
                  f"{result['status']} "
                  f"wall={result['wall_seconds']}s makespan={result.get('makespan')} valid={result.get('valid')} "
                  f"sat={model.get('total_solving_time')}s variables={model.get('variables')} "
                  f"clauses={model.get('clauses')} "
                  f"stage={result.get('timeout_stage', result['stage'])} "
                  f"stop={result.get('stop_reason')}", flush=True)
            write_summary(folder, results)
            if args.compare:
                (folder / "comparison.md").write_text(
                    comparison_markdown(args.dataset_dir, files, args.solvers, results))
    summary = write_summary(folder, results)
    print(f"Complete: {json.dumps(summary['statuses'])}", flush=True)
    if args.compare:
        table = comparison_markdown(args.dataset_dir, files, args.solvers, results)
        (folder / "comparison.md").write_text(table)
        print("\n" + table, flush=True)


if __name__ == "__main__":
    main()
