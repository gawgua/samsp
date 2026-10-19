"""SAMSP res-uni: unrestricted MFS, separate BDDs, strengthened adders.

Coelho and Vanhoucke (2026), sections 3.2 and 4.3.3. Boolean constants use
Python bools, never DIMACS literals. Each resource row commits atomically.
The paper's final variable/clause ceilings are deliberately omitted; the
caller supplies a wall deadline and enforces combined process RSS externally.
"""

from dataclasses import dataclass
from itertools import combinations, product
import time


class EncodingLimitError(TimeoutError):
    stop_reason = "time_limit"


class _StageLimit(Exception):
    pass


@dataclass(frozen=True)
class ConversionLimits:
    stage_seconds: float = 1.0
    mfs_clauses: int = 1_000_000


def negate(literal):
    return not literal if type(literal) is bool else -literal


def normalize_row(literals, bound, weights=None):
    """Normalize explicit constants, repeated and complementary literals."""
    literals = list(literals)
    weights = [1] * len(literals) if weights is None else list(weights)
    if len(literals) != len(weights):
        raise ValueError("Literal/weight lengths differ")
    coefficients = {}
    for literal, weight in zip(literals, weights):
        if type(weight) is not int or weight < 0:
            raise ValueError("Weights must be nonnegative integers")
        if type(literal) is bool:
            if literal:
                bound -= weight
        elif type(literal) is not int or literal == 0:
            raise ValueError("Expected a nonzero DIMACS literal or bool")
        else:
            coefficients[literal] = coefficients.get(literal, 0) + weight
    for literal in list(coefficients):
        if literal > 0 and -literal in coefficients:
            common = min(coefficients[literal], coefficients[-literal])
            bound -= common
            coefficients[literal] -= common
            coefficients[-literal] -= common
    return {literal: weight for literal, weight in coefficients.items() if weight}, bound


class ClauseBuffer:
    """Private row allocation. Only a completed row advances the WCNF top ID."""
    def __init__(self, top_id, guard=None, exclusive=None):
        self.nv = top_id
        self.clauses = []
        self.guard = guard
        self.exclusive = exclusive or (lambda a, b: False)
        self.supports = {}
        self.omitted_mfs_clauses = 0
        self.omitted_adder_clauses = 0
        self.simplified_carries = 0
        self.operations = 0

    def check(self):
        if self.guard is not None:
            self.guard(self)

    def new(self):
        self.nv += 1
        self.check()
        return self.nv

    def clause(self, literals):
        # Filter bools before sets: True == 1 and False == 0 in Python.
        values = set()
        for literal in literals:
            if type(literal) is bool:
                if literal:
                    return
                continue
            if -literal in values:
                return
            values.add(literal)
        self.clauses.append(sorted(values, key=lambda lit: (abs(lit), lit)))
        if len(self.clauses) % 32 == 0:
            self.check()

    def mutually_exclusive(self, a, b):
        left = self.supports.get(a, frozenset((a,)))
        right = self.supports.get(b, frozenset((b,)))
        return all(self.exclusive(x, y) for x in left for y in right)


def _trivial(buffer, coefficients, bound):
    buffer.check()
    if bound < 0:
        buffer.clause([])
        return True
    return bound >= sum(coefficients.values())


def _forbidden(buffer, chosen):
    buffer.operations += 1
    if buffer.operations % 32 == 0:
        buffer.check()
    if any(buffer.exclusive(a, b) for a, b in combinations(chosen, 2)):
        buffer.omitted_mfs_clauses += 1
    else:
        buffer.clause([-literal for literal in chosen])


def encode_mfs(buffer, coefficients, bound):
    """Enumerate every minimal overload; omit only temporally impossible sets."""
    if _trivial(buffer, coefficients, bound):
        return
    items = list(coefficients.items())
    if all(weight == 1 for _, weight in items):
        for forbidden in combinations(coefficients, bound + 1):
            _forbidden(buffer, forbidden)
        buffer.check()
        return
    suffix = [0] * (len(items) + 1)
    for index in range(len(items) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + items[index][1]
    stack = [(0, (), 0, None)]
    while stack:
        buffer.check()
        index, chosen, total, smallest = stack.pop()
        if total > bound:
            if total - smallest <= bound:
                _forbidden(buffer, chosen)
        elif index < len(items) and total + suffix[index] > bound:
            literal, weight = items[index]
            stack.append((index + 1, chosen, total, smallest))
            stack.append((index + 1, (*chosen, literal), total + weight,
                          weight if smallest is None else min(smallest, weight)))
    buffer.check()


def encode_bdd(buffer, coefficients, bound):
    """Separate reduced resource BDD with six-clause equivalence ITE gates."""
    if _trivial(buffer, coefficients, bound):
        return
    items = list(coefficients.items())
    suffix = [0] * (len(items) + 1)
    for index in range(len(items) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + items[index][1]
    states, shared = {}, {}
    stack = [(0, bound, False)]
    while stack:
        buffer.check()
        index, remaining, ready = stack.pop()
        key = (index, remaining)
        if key in states:
            continue
        if remaining < 0:
            states[key] = False
        elif remaining >= suffix[index]:
            states[key] = True
        elif not ready:
            stack.append((index, remaining, True))
            stack.append((index + 1, remaining, False))
            stack.append((index + 1, remaining - items[index][1], False))
        else:
            x, weight = items[index]
            high, low = states[index + 1, remaining - weight], states[index + 1, remaining]
            # bool constants cannot alias auxiliaries: auxiliaries exceed inputs.
            if high == low and type(high) is type(low):
                states[key] = high
                continue
            gate = (x, (type(high), high), (type(low), low))
            if gate not in shared:
                p = buffer.new()
                for clause in ((-x, negate(high), p), (x, negate(low), p),
                               (-x, high, -p), (x, low, -p),
                               (negate(high), negate(low), p), (high, low, -p)):
                    buffer.clause(clause)
                shared[gate] = p
            states[key] = shared[gate]
    buffer.clause([states[0, bound]])
    buffer.check()


def _adder_gate(buffer, inputs):
    result = buffer.new()
    support = frozenset().union(*(buffer.supports.get(x, frozenset((x,))) for x in inputs))
    buffer.supports[result] = support
    if all(buffer.mutually_exclusive(a, b) for a, b in combinations(inputs, 2)):
        # Under the certified AMO premise, XOR equals OR and carry is false.
        for literal in inputs:
            buffer.clause([-literal, result])
        buffer.clause([-result, *inputs])
        buffer.simplified_carries += 1
        buffer.omitted_adder_clauses += (9 if len(inputs) == 2 else 20) - (len(inputs) + 1)
        return result, False
    carry = buffer.new()
    buffer.supports[carry] = support
    for bits in product((False, True), repeat=len(inputs)):
        buffer.clause([-literal if bit else literal for literal, bit in zip(inputs, bits)]
                      + [result if sum(bits) % 2 else -result])
    if len(inputs) == 2:
        a, b = inputs
        for clause in ((-carry, a), (-carry, b), (carry, -a, -b)):
            buffer.clause(clause)
    else:
        for a, b in combinations(inputs, 2):
            buffer.clause([-a, -b, carry])
            buffer.clause([a, b, -carry])
    # Strengthening clauses from section 3.2.3 (res-add(ext)).
    for literal in inputs:
        buffer.clause([carry, result, -literal])
        if len(inputs) == 3:
            buffer.clause([-carry, -result, literal])
    return result, carry


def encode_adder(buffer, coefficients, bound):
    """Strengthened half/full adders, safe precedence carries, binary comparison."""
    if _trivial(buffer, coefficients, bound):
        return
    columns = []
    for literal, weight in coefficients.items():
        for bit in range(weight.bit_length()):
            if weight & (1 << bit):
                while len(columns) <= bit:
                    columns.append([])
                columns[bit].append(literal)
    index = 0
    while index < len(columns):
        buffer.check()
        while len(columns[index]) > 1:
            count = min(3, len(columns[index]))
            inputs = [columns[index].pop() for _ in range(count)]
            result, carry = _adder_gate(buffer, inputs)
            columns[index].append(result)
            if carry is not False:
                if index + 1 == len(columns):
                    columns.append([])
                columns[index + 1].append(carry)
        index += 1
    bits = [column[0] if column else False for column in columns]
    # Certified zero carries can shorten the sum below the capacity bit width.
    bits.extend([False] * max(0, bound.bit_length() - len(bits)))
    for index, literal in enumerate(bits):
        if not bound & (1 << index):
            buffer.clause([negate(literal)] + [negate(bits[higher]) for higher in range(index + 1, len(bits))
                                              if bound & (1 << higher)])
    buffer.check()


KERNELS = dict(mfs=encode_mfs, bdd=encode_bdd, adder=encode_adder)


class UnifiedRenewableEncoder:
    """Chronological staged conversion, with atomic row retries and a deadline."""
    def __init__(self, formula, stage="mfs", limits=None, clock=None, deadline=None):
        if stage not in KERNELS:
            raise ValueError("Unknown renewable stage")
        self.formula = formula
        self.limits = limits or ConversionLimits()
        self.clock = clock or time.monotonic
        self.deadline = deadline
        self.stage = stage
        self.stages = []
        self.rows = 0
        self.complete = False
        self.total_started = self.clock()
        self._start_stage()

    def _start_stage(self):
        self.started = self.clock()
        self.current = dict(stage=self.stage, rows=0, clauses=0, auxiliary_variables=0,
                            discarded_rows=0, discarded_clauses=0, discarded_variables=0,
                            omitted_mfs_clauses=0, omitted_adder_clauses=0,
                            simplified_carries=0, seconds=0.0, switch_reason=None)
        self.stages.append(self.current)

    def check_limits(self, buffer=None, *, stage_budget=True):
        now = self.clock()
        if self.deadline is not None and now >= self.deadline:
            raise EncodingLimitError("Renewable conversion wall deadline exhausted")
        extra = 0 if buffer is None else len(buffer.clauses)
        clauses = len(self.formula.hard) + len(self.formula.soft) + extra
        if stage_budget:
            if self.stage == "mfs" and clauses >= self.limits.mfs_clauses:
                raise _StageLimit("mfs_clause_limit")
            if self.stage != "adder" and now - self.started >= self.limits.stage_seconds:
                raise _StageLimit("stage_time_limit")

    def _discard(self, buffer):
        if buffer is not None:
            self.current["discarded_rows"] += 1
            self.current["discarded_clauses"] += len(buffer.clauses)
            self.current["discarded_variables"] += buffer.nv - self.formula.nv

    def _switch(self, reason, buffer=None):
        self.current["seconds"] = self.clock() - self.started
        self.current["switch_reason"] = reason
        self._discard(buffer)
        self.stage = "bdd" if self.stage == "mfs" else "adder"
        self._start_stage()

    def begin_row(self):
        while True:
            try:
                self.check_limits()
                return
            except _StageLimit as error:
                self._switch(str(error))

    def add_row(self, coefficients, bound, exclusive=None):
        while True:
            buffer = ClauseBuffer(self.formula.nv, self.check_limits, exclusive)
            try:
                self.check_limits(buffer)
                KERNELS[self.stage](buffer, coefficients, bound)
                self.check_limits(buffer)
                top = self.formula.nv
                self.formula.hard.extend(buffer.clauses)
                self.formula.nv = buffer.nv
                self.rows += 1
                self.current["rows"] += 1
                self.current["clauses"] += len(buffer.clauses)
                self.current["auxiliary_variables"] += buffer.nv - top
                for key in ("omitted_mfs_clauses", "omitted_adder_clauses", "simplified_carries"):
                    self.current[key] += getattr(buffer, key)
                return
            except _StageLimit as error:
                self._switch(str(error), buffer)
            except EncodingLimitError:
                self._discard(buffer)
                raise

    def finish(self, expected_rows):
        self.check_limits(stage_budget=False)
        if self.rows != expected_rows:
            raise EncodingLimitError(f"Incomplete renewable conversion: {self.rows}/{expected_rows} rows")
        self.complete = True
        self.current["seconds"] = self.clock() - self.started
        self.seconds = self.clock() - self.total_started

    def statistics(self):
        stages = [dict(stage) for stage in self.stages]
        if not self.complete:
            stages[-1]["seconds"] = self.clock() - self.started
        keys = ("clauses", "auxiliary_variables", "omitted_mfs_clauses", "omitted_adder_clauses", "simplified_carries")
        return dict(final_stage=self.stage, rows=self.rows, complete=self.complete, stages=stages,
                    seconds=self.seconds if self.complete else self.clock() - self.total_started,
                    **{key: sum(stage[key] for stage in stages) for key in keys})
