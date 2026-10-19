from solver.solver import SOLVER_STATUS
from dataset.problems import SAMSPProblem
from threading import Event, Timer
import time
import numpy as np

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.pb import PBEnc, EncType
from solver.renewable_res_uni import UnifiedRenewableEncoder, EncodingLimitError

class SAMSPMaxSAT_CPM2_Solver:
    """CPM2 hard encoding with a sink-only weighted MaxSAT objective."""

    def __init__(self, problem: SAMSPProblem, LB=None, UB=None, renewable_encoding="amo"):
        if renewable_encoding not in {"amo", "res_uni"}:
            raise ValueError("renewable_encoding must be amo or res_uni")
        self.renewable_encoding = renewable_encoding
        self.__renewable_encoder = None
        self.__renewable_statistics = {}
        self.__build_attempted = False
        self.__encoding_deadline = None
        self.__encoding_started = None
        self.__phase_ranges = {}
        self.optimization = "maxsat"
        self.__formula = WCNF()
        self.__model_built = False
        self.__problem = problem
        self.__duration_values = tuple(map(int, problem.durations))
        self.__renewable_requests = np.asarray(problem.renewable_requests, dtype=np.int64)
        self.__cumulative_requests = np.asarray(problem.cumulative_requests, dtype=np.int64)
        self.__ES, self.__LS, self.__EC, self.__LC = (
            np.asarray(getattr(problem, name), dtype=np.int64) for name in ('ES', 'LS', 'EC', 'LC'))
        if hasattr(problem, 'precedence_arcs'):
            self.__precedence_arcs = problem.precedence_arcs
        else:
            # Preserve compatibility with callers supplying the graph parser.
            self.__precedence_arcs = np.asarray(
                [(a, b, data.get('weight', 0)) for a, b, data in problem.precedence_graph.edges(data=True)],
                dtype=np.int64).reshape((-1, 3))
        self.__LB = int(max(problem.LB, problem.ES[problem.sink_id], LB if LB is not None else problem.LB))
        self.__UB = int(UB if UB is not None and UB < problem.UB else problem.UB)
        self.__model = None
        self.__maxsat_statistics = {}
        self.__rc2 = None
        self.__maxsat_calls = 0
        self.__encoding_seconds = None
        self.__optimization_seconds = None
        self.__optimization_started = None
        self.objective_cost = None

        self.__preprocessing()

    def create_new_variable(self):
        self.__formula.nv += 1
        return self.__formula.nv

    def add_clause(self, clause):
        self.__formula.append(clause)

    def encode(self, deadline=None):
        """Compatibility entry point for the benchmark runner."""
        self.build_model(deadline=deadline)

    def build_model(self, deadline=None):
        """Construct all hard clauses and the weighted soft objective once."""
        if self.__model_built:
            return
        if self.__build_attempted:
            raise RuntimeError("Incomplete model cannot be rebuilt; create a fresh solver")
        self.__build_attempted = True
        self.__encoding_deadline = deadline
        started = time.monotonic()
        self.__encoding_started = started
        try:
            self.__build_hard_model()
            self.__build_objective()
            self.__model_built = True
        finally:
            self.__encoding_seconds = time.monotonic() - started
            self.__encoding_started = None

    def __build_hard_model(self):
        phases = (("timing", (self.__consistency_clause, self.__start_clauses,
                               self.__precedence_constraints)),
                  ("renewable", (self.__renewable_resources_constraints,)),
                  ("storage", (self.__cummulative_resources_constraints,)))
        for name, methods in phases:
            if self.__encoding_deadline is not None and time.monotonic() >= self.__encoding_deadline:
                raise EncodingLimitError("Model construction wall deadline exhausted")
            first = len(self.__formula.hard)
            for method in methods:
                method()
            self.__phase_ranges[name] = (first, len(self.__formula.hard))

    def __build_objective(self):
        sink = self.__problem.sink_id
        if self.__LB > self.__UB:
            self.add_clause([])
        else:
            self.add_clause([self.L[sink, self.__UB]])
            if self.__LB > self.__problem.ES[sink]:
                self.add_clause([-self.L[sink, self.__LB - 1]])
        for t in range(self.__LB, self.__UB):
            self.__formula.append([self.L[sink, t]], weight=1)

    def get_hard_clause_phases(self):
        """Clause boundaries for timing/storage parity checks; no clause copies."""
        return dict(self.__phase_ranges)

    def get_hard_clauses(self):
        """Return canonical clauses for comparisons, retaining duplicates."""
        return sorted((tuple(sorted(clause)) for clause in self.__formula.hard),
                      key=lambda clause: (len(clause), clause))

    def get_soft_clauses(self):
        return self.__formula.soft

    def solve(self, time_limit=None):
        if self.__build_attempted and not self.__model_built:
            return SOLVER_STATUS.UNKNOWN
        if time_limit is not None and time_limit <= 0:
            return SOLVER_STATUS.UNKNOWN
        started = time.monotonic()
        self.__optimization_started = started
        deadline = None if time_limit is None else started + time_limit
        try:
            status = self.__solve_maxsat(deadline)
            if status == SOLVER_STATUS.OPTIMAL:
                self.objective_cost = self.get_makespan() - self.__LB
            return status
        finally:
            self.__optimization_seconds = time.monotonic() - started
            self.__optimization_started = None

    def __solve_maxsat(self, deadline):
        if not self.__model_built:
            raise RuntimeError("Call build_model() or encode() before solve()")
        interrupted = Event()
        with RC2(self.__formula, solver="g4", incr=True) as rc2:
            self.__rc2 = rc2
            self.__maxsat_calls = 0
            call_oracle = rc2._call_oracle

            def counted_call(*args, **kwargs):
                self.__maxsat_calls += 1
                return call_oracle(*args, **kwargs)

            rc2._call_oracle = counted_call
            timer = None
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self.__rc2 = None
                return SOLVER_STATUS.UNKNOWN

            def interrupt():
                interrupted.set()
                rc2.interrupt()

            if remaining is not None:
                timer = Timer(remaining, interrupt)
                timer.start()
            try:
                model = rc2.compute(expect_interrupt=True)
                if interrupted.is_set() or rc2.interrupted:
                    return SOLVER_STATUS.UNKNOWN
                if model is None:
                    return SOLVER_STATUS.UNSATISFIABLE
                self.__model = model
                makespan = self.get_makespan()
                if makespan is None or rc2.cost != makespan - self.__LB:
                    raise ValueError("MaxSAT objective cost does not match decoded makespan")
                self.objective_cost = rc2.cost
                return SOLVER_STATUS.OPTIMAL
            finally:
                if timer is not None:
                    timer.cancel()
                    timer.join()
                self.__maxsat_statistics = {
                    "number_of_calls": self.__maxsat_calls,
                    "total_solving_time": rc2.oracle.time_accum(),
                    "native_statistics": rc2.oracle.accum_stats(),
                    "rc2_cost": rc2.cost,
                }
                self.__rc2 = None

    def get_statistics(self):
        result = {"variables": self.__formula.nv, "clauses": len(self.__formula.hard),
                  "number_of_calls": 0, "total_solving_time": 0}
        result.update(self.__maxsat_statistics)
        rc2 = self.__rc2
        if rc2 is not None and rc2.oracle is not None:
            result.update(total_solving_time=rc2.oracle.time_accum(),
                          native_statistics=rc2.oracle.accum_stats(), rc2_cost=rc2.cost,
                          number_of_calls=self.__maxsat_calls)
        result.update(optimization=self.optimization, hard_clauses=result["clauses"],
                      soft_clauses=len(self.__formula.soft),
                      objective_cost=self.objective_cost, objective_lower_bound=self.__LB,
                      objective_upper_bound=self.__UB,
                      encoding_seconds=(time.monotonic() - self.__encoding_started
                                        if self.__encoding_started is not None else self.__encoding_seconds),
                      optimization_seconds=(time.monotonic() - self.__optimization_started
                                            if self.__optimization_started is not None
                                            else self.__optimization_seconds))
        result.update(renewable_encoding=self.renewable_encoding, model_complete=self.__model_built,
                      renewable=(self.__renewable_encoder.statistics() if self.__renewable_encoder is not None
                                 else dict(self.__renewable_statistics)))
        return result

    def get_last_feasible_model(self):
        return self.__model

    def get_makespan(self):
        model = self.get_last_feasible_model()
        if model is None:
            return None
        sink = self.__problem.sink_id
        for t in range(self.__problem.ES[sink], self.__problem.UB + 1):
            if model[self.L[sink, t] - 1] > 0:
                return t
        return None

    def __preprocessing(self):
        # Auxiliary variables
        self.u = {}  # u(ai,t) activity i is running at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__problem.ES[i], self.__problem.LC[i] + 1):
                self.u[i, t] = self.create_new_variable()
        self.L = {}  # L(ai,t) activity i is started before or at time t
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__problem.ES[i], self.__problem.UB + 1):
                self.L[i, t] = self.create_new_variable()
        self.register = {}  # R(Ai, s, t) activity i is started in time range [s, t]

    def __consistency_clause(self):
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__problem.ES[i], self.__problem.UB):
                self.add_clause([-self.L[i, t], self.L[i, t + 1]])

        for i in range(self.__problem.number_of_activities):
            earliest = int(self.__problem.ES[i])
            duration = self.__duration_values[i]
            for t in range(earliest, self.__problem.LC[i]):
                self.add_clause([-self.u[i, t], self.L[i, t]])
                previous_time = t - duration
                if previous_time < earliest:
                    # Before ES the activity has not started, so L is false.
                    self.add_clause([self.u[i, t], -self.L[i, t]])
                else:
                    self.add_clause([-self.u[i, t], -self.L[i, previous_time]])
                    self.add_clause([self.u[i, t], -self.L[i, t], self.L[i, previous_time]])
                  
    def __start_clauses(self):
        for i in range(self.__problem.number_of_activities):
            self.add_clause([self.L[i, self.__problem.LS[i]]])

    def __precedence_constraints(self):
        for arc in self.__precedence_arcs:
            predecessor, successor, weight = map(int, arc)
            predecessor_times = range(int(self.__problem.ES[predecessor]), int(self.__problem.LS[predecessor]) + 1)
            for t in range(self.__problem.ES[successor], self.__problem.LS[successor] + 1):
                if t - weight in predecessor_times:
                    self.add_clause(
                        [-self.L[successor, t], self.L[predecessor, t - weight]]
                    )

    def __renewable_resources_constraints(self):
        if self.renewable_encoding == "res_uni":
            self.__res_uni_constraints()
            return
        started, clauses, variables = time.monotonic(), len(self.__formula.hard), self.__formula.nv
        for r in range(self.__problem.number_of_renewable_resources):
            for t in range(self.__UB):
                active = np.flatnonzero((self.__renewable_requests[:, r] > 0)
                                        & (self.__ES <= t) & (t < self.__LC))
                activities_using_r_at_t = [self.u[int(i), t] for i in active]

                if len(activities_using_r_at_t) > 1:
                    # AMO for activities using resource r at time t
                    R = {}
                    R[0] = self.create_new_variable()
                    self.add_clause([-activities_using_r_at_t[0], R[0]])
                    for k in range(1, len(activities_using_r_at_t) - 1):
                        R[k] = self.create_new_variable()
                        self.add_clause([-activities_using_r_at_t[k], R[k]])
                        self.add_clause([-R[k - 1], R[k]])

                        self.add_clause([-R[k - 1], -activities_using_r_at_t[k]])

                    self.add_clause(
                        [
                            -R[len(activities_using_r_at_t) - 2],
                            -activities_using_r_at_t[-1],
                        ]
                    )
        self.__renewable_statistics = dict(
            rows=self.__UB * self.__problem.number_of_renewable_resources, complete=True,
            clauses=len(self.__formula.hard) - clauses, auxiliary_variables=self.__formula.nv - variables,
            seconds=time.monotonic() - started, stages=[])

    def __res_uni_constraints(self):
        encoder = UnifiedRenewableEncoder(self.__formula, deadline=self.__encoding_deadline)
        self.__renewable_encoder = encoder
        closure = self.__problem.temporal_closure
        durations = np.asarray(self.__duration_values)
        # A generalized lag implies non-overlap only if it spans the full duration.
        nonoverlap = (closure >= durations[:, None])
        nonoverlap |= nonoverlap.T.copy()
        np.fill_diagonal(nonoverlap, False)
        for t in range(self.__UB):
            for r in range(self.__problem.number_of_renewable_resources):
                encoder.begin_row()
                active = np.flatnonzero((self.__renewable_requests[:, r] > 0)
                                        & (durations > 0) & (self.__ES <= t) & (t < self.__LC))
                activities = {self.u[int(i), t]: int(i) for i in active}
                coefficients = {literal: int(self.__renewable_requests[i, r])
                                for literal, i in activities.items()}
                def exclusive(a, b):
                    return a in activities and b in activities and bool(nonoverlap[activities[a], activities[b]])
                encoder.add_row(coefficients, int(self.__problem.renewable_capacities[r]), exclusive)
        encoder.finish(self.__UB * self.__problem.number_of_renewable_resources)

    def __cummulative_resources_constraints(self):
        # usage at t of r <= capacity of r - ps of r
        for r in range(self.__problem.number_of_cumulative_resources):
            for t in range(self.__UB + 1):
                lits = []
                weights = []
                limit = int(self.__problem.cumulative_capacities[r] - self.__problem.PS[r, t])

                deltas = self.__cumulative_requests[:, r]
                active = np.flatnonzero(((deltas > 0) & (self.__ES <= t) & (t < self.__LS))
                                        | ((deltas < 0) & (self.__EC <= t) & (t < self.__LC)))
                for item in active:
                    i = int(item)
                    c = int(deltas[i])
                    if c > 0:
                        lits.append(self.L[i, t])
                        weights.append(c)
                    else:
                        w = -c
                        t_finish = t - self.__duration_values[i]
                        lits.append(-self.L[i, t_finish])
                        weights.append(w)
                        limit += w

                if limit < 0:
                    raise ValueError(
                        f"Negative limit for cumulative resource {r} at time {t}: {limit}"
                    )

                if lits:
                    clauses = PBEnc.leq(
                        lits=lits, weights=weights, bound=limit,
                        top_id=self.__formula.nv, encoding=EncType.seqcounter,
                    ).clauses
                    for clause in clauses:
                        self.add_clause(clause)
