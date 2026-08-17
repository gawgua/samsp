from enum import Enum, auto
from threading import Timer
from typing import Iterable

from pysat.pb import PBEnc, EncType
from pysat.solvers import Glucose4


class SOLVER_STATUS(Enum):
    UNSATISFIABLE = auto()
    OPTIMAL = auto()
    SATISFIABLE = auto()
    UNKNOWN = auto()


class SATSolver:
    def __init__(self):
        self.__number_of_variables = 0
        self.__solver = None
        self.__assumption = set()
        self.__last_feasible_model = None
        self.__number_of_calls = 0
        self.__temp_clauses = set()

    def create_new_variable(self) -> int:
        self.__number_of_variables += 1
        return self.__number_of_variables

    def add_clause(self, clause: Iterable[int]):
        self.__temp_clauses.add(tuple(sorted(clause)))

    def solve(self, time_limit=None) -> bool | None:
        if time_limit is not None and time_limit <= 0:
            raise ValueError("Time limit must be greater than 0.")
        if self.__solver is None:
            self.__temp_clauses = sorted(self.__temp_clauses, key=lambda t: (len(t), t))
            self.__solver = Glucose4(
                bootstrap_with=self.__temp_clauses, use_timer=True, incr=True
            )
        self.__number_of_calls += 1
        self.__solver.clear_interrupt()
        timer = None
        if time_limit is not None:

            def interrupt(s):
                s.interrupt()

            timer = Timer(time_limit, interrupt, [self.__solver])
            timer.start()
        try:
            result = self.__solver.solve_limited(
                expect_interrupt=True, assumptions=self.__assumption
            )
            if result is not None and result:
                self.__last_feasible_model = self.__solver.get_model()
            return result
        finally:
            if timer:
                timer.cancel()

    def get_last_feasible_model(self) -> list[int] | None:
        return self.__last_feasible_model

    def get_model(self) -> list[int] | None:
        return self.__solver.get_model()

    def add_assumption(self, assumption: int | Iterable[int]):
        if isinstance(assumption, int):
            self.__assumption.add(assumption)
        else:
            self.__assumption.update(assumption)

    def clear_assumptions(self):
        self.__assumption.clear()

    def clear_interrupt(self):
        self.__solver.clear_interrupt()

    def get_statistics(self) -> dict[str, int | float]:
        result = {
            "variables": self.__number_of_variables,
            "clauses": len(self.__temp_clauses),
            "number_of_calls": self.__number_of_calls,
        }
        if self.__solver is not None:
            result["total_solving_time"] = self.__solver.time_accum()
        else:
            result["total_solving_time"] = 0
        return result

    def add_at_most_k(self, literals: list[int], weights: list[int], k: int):
        cnf = PBEnc.leq(
            lits=literals,
            weights=weights,
            bound=k,
            top_id=self.__number_of_variables,
            encoding=EncType.seqcounter,
        ).clauses
        if not cnf:
            return
        new_variable_max_index = -1
        for clause in cnf:
            for var in clause:
                if abs(var) > new_variable_max_index:
                    new_variable_max_index = abs(var)
        if new_variable_max_index == -1:
            return
        self.__number_of_variables = max(
            new_variable_max_index, self.__number_of_variables
        )
        for clause in cnf:
            self.__temp_clauses.add(tuple(sorted(clause)))
