"""Solve SAMSP by translating integer start times into weighted MaxSAT.

Reading guide
-------------
Read the notation below, then __init__, build_model, solve, and _compute.
The smaller methods implement reusable Boolean and resource encodings.
No CP result, saved schedule, or instance-specific answer is consumed.

Terminology used in the comments
-------------------------------
SAT asks whether Boolean clauses can all be satisfied. CNF (conjunctive
normal form) is an AND of OR clauses. MaxSAT minimizes the weight of
violated soft clauses while satisfying every hard clause. WCNF means
weighted CNF. A PB (pseudo-Boolean) constraint is an integer-weighted sum
of Boolean values. A cardinality constraint has only unit weights; AMO
(at most one) is its bound-one case. A BDD is a binary decision diagram
used here to translate weighted constraints into CNF. RC2 is PySAT's
core-guided MaxSAT implementation: it uses unsatisfiable cores (conflicting
subsets of assumptions) to refine the soft-clause optimization. The AIG
(AND-inverter graph) import proxy below serves an optional converter only.
FIFO means first in, first out; here it describes certified ordering of
storage acquisition and release events, not a queue implemented in Python.

Scheduling notation and input fields
------------------------------------
S[i] is activity i's nonnegative integer start; d[i] is its duration.
The source starts at zero. The sink starts after every activity finishes;
its start M is therefore the makespan to minimize. A lag (a, b, w) means
S[b] − S[a] ≥ w. Negative lag weights can express maximum delays.

``closure[a, b]`` is the strongest known lower bound on S[b] − S[a].
It is a max-plus temporal closure, not a Boolean literal table. A positive
cycle is inconsistent; closure helpers return None on detected conflicts.
ES/LS mean earliest/latest start; EC/LC mean earliest/latest completion.
These arrays live on ``problem`` and are updated for each trial horizon.
Local ``p`` is a shorter reference to ``problem``. Indices i, a, b, g, e
identify activities; r identifies a resource or storage zone in its own
loop; t is an integer timestamp. g/e specifically denote acquisition/release
activities. ``jobs`` is a selected subset of activity IDs. A "row" means
one capacity inequality for one resource and one timestamp, not a data-file
row. An auxiliary variable helps encode a constraint; it is not an activity.
A domain is the allowed start-time set for one activity. A certificate is a
proved condition that permits a shortcut; a witness records the concrete
subset and numbers supporting a bound. Resource energy is duration times
demand, summed over selected jobs. A release bound r means those jobs cannot
start before r. A tail bound q means each job finishes at least q time units
before the sink. Their work must fit between r and M − q. Saturation means
that interval has exactly enough capacity for the selected work and no slack.
A cache stores previously created literals or rows so they can be reused.

``renewable_requests[i][r]`` is the amount of renewable resource r used
during [S[i], S[i] + d[i]). ``cumulative_requests[i][r]`` is historical
naming for a signed STORAGE event: a positive value acquires positions at
S[i]; a negative value releases positions at S[i] + d[i]. Storage usage
must stay between zero and cumulative_capacities[r], including at the
final time boundary. It can persist while no activity is executing.

Boolean notation and PySAT representation
-----------------------------------------
L[i,t] means S[i] ≤ t. It is an order predicate, not an exact-start
predicate. If S[i] = 3, its values at t = 1, 2, 3, 4 are false, false,
true, true. The name L here must not be confused with ``closure``.
Execution is L[i,t] ∧ ¬L[i,t−d[i]]. A certified storage pair (g,e)
occupies a position when L[g,t] ∧ ¬L[e,t−d[e]].

PySAT uses positive integers for Boolean variables and negative integers
for their negations. A list is one OR clause; the hard clause list is an
AND of clauses. For example, [-a, b] says that a implies b. The empty
clause [] is always false and makes the hard formula unsatisfiable.
Variable 1 is reserved and forced true, so literal 1 means TRUE and -1
means FALSE everywhere in this class. Fresh decision variables start at 2.

WCNF holds hard clauses, which must all hold, and weighted soft clauses,
whose violated weights RC2 minimizes. One unit-weight soft clause
[L[sink,t]] is added for each t from lower_bound through horizon − 1.
A schedule with makespan M violates exactly M − lower_bound of them.
For example, lower_bound = 5 and M = 8 give objective_cost = 3.

Bounds, optimization, and lifecycle
----------------------------------
``lower_bound`` is a certified floor, not a feasible schedule. ``horizon``
is the current finite search limit, not a certified incumbent, even though
the input field is named UB. The initial horizon is a feasibility trial.
If its hard formula is infeasible, solve expands the horizon and rebuilds.
Once RC2 finds a MaxSAT optimum inside a feasible horizon, it is globally
optimal: every schedule with a smaller makespan also fits that horizon.
A timeout returns UNKNOWN, never an infeasibility or optimality proof.
Unless preprocessing detects global inconsistency, repeated infeasible
horizons alone do not establish global infeasibility; use a time limit.

Typical call sequence: construct SAMSPMaxSATOptimizedSolver(problem), call
build_model(), then solve(time_limit=...), then inspect get_makespan(),
get_last_feasible_model(), and get_statistics(). solve() expects the first
model to have been built by the caller. The returned Boolean model is not
a start-time dictionary; decode each activity's first true order predicate
and validate the resulting schedule independently.

For an already loaded problem, the essential usage is:

    solver = SAMSPMaxSATOptimizedSolver(problem)
    solver.build_model()
    status = solver.solve(time_limit=30)
    if status == SOLVER_STATUS.OPTIMAL:
        model = solver.get_last_feasible_model()
        schedule = {
            i: next(t for t in range(problem.ES[i], problem.LS[i] + 1)
                    if model[solver.get_L(i, t) - 1] > 0)
            for i in problem.activity_ids
        }

``schedule[i]`` is then the decoded start, whereas ``model`` contains signed
Boolean assignments, including auxiliary variables. On UNKNOWN, this code
does not assume a new feasible schedule has been retained.
"""

# bisect_right maps a threshold in a sparse domain to the preceding allowed
# start. Event communicates interruption between threads; Timer schedules it.
# monotonic time measures elapsed duration without wall-clock adjustments.
from bisect import bisect_right
from threading import Event, Timer
import time

# PySAT eagerly imports its optional AIG converter, although this backend uses
# only directly constructed clauses. Defer that package until an AIG API is
# actually requested. The proxy preserves conversion support and changes no
# formula or solving semantics.
import importlib
import importlib.util
import sys
from types import ModuleType

# Install the lazy proxy only before either package has been imported. Existing
# module objects must retain their identity for code that already references them.
if 'pysat.formula' not in sys.modules and 'aiger_cnf' not in sys.modules:
    _aig_spec = importlib.util.find_spec('aiger_cnf')
    # find_spec locates optional package metadata without loading its body. If the
    # package is absent, leave normal PySAT import handling in control.
    if _aig_spec is not None:
        # Copy normal module/package metadata so introspection and submodule imports
        # can still recognize the proxy as aiger_cnf.
        _aig_proxy = ModuleType('aiger_cnf')
        _aig_proxy.__spec__ = _aig_spec
        _aig_proxy.__loader__ = _aig_spec.loader
        _aig_proxy.__package__ = 'aiger_cnf'
        _aig_proxy.__file__ = _aig_spec.origin
        if _aig_spec.submodule_search_locations is not None:
            _aig_proxy.__path__ = _aig_spec.submodule_search_locations
        _aig_real = None

        def _load_aig_attribute(name):
            """Load the optional AIG converter on its first attribute access.

            ``name`` is requested by Python's module attribute lookup. Replace the
            proxy in sys.modules with the real package once, then return that package's
            attribute. If import fails, restore the proxy and re-raise the original
            exception so a later lookup can retry. This affects import timing only;
            this solver itself constructs CNF directly and never needs AIG conversion.
            """
            global _aig_real
            if _aig_real is None:
                # Remove the proxy first; otherwise import_module would return this same
                # proxy instead of executing the real package.
                sys.modules.pop('aiger_cnf', None)
                try:
                    _aig_real = importlib.import_module('aiger_cnf')
                except BaseException:
                    sys.modules['aiger_cnf'] = _aig_proxy
                    raise
            return getattr(_aig_real, name)

        # Python calls module.__getattr__ only for attributes absent on the proxy.
        # Registering the proxy makes PySAT's eager import cheap until such a lookup.
        _aig_proxy.__getattr__ = _load_aig_attribute
        sys.modules['aiger_cnf'] = _aig_proxy

# CardEnc handles unweighted counts; PBEnc handles weighted sums. RC2 is the
# core-guided weighted MaxSAT optimizer. WCNF stores hard and soft clauses.
# The two EncType classes are separate enums, hence the CardType/PBType aliases.
from pysat.card import CardEnc, EncType as CardType
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.pb import PBEnc, EncType as PBType

# Resource helpers derive necessary orders and workload bounds. _closure adds
# structural timing constraints to a private matrix; _add_arc updates that matrix.
# SOLVER_STATUS supplies the result enum shared with the repository's runners.
from solver.resource_bounds import disjunctive_resource_closure, resource_lower_bound, strengthen_resource_lags
from solver.schedule_upper_bound import _add_arc, _closure
from solver.solver import SOLVER_STATUS


class SAMSPMaxSATOptimizedSolver:
    """Own preprocessing, one bounded formula, and the MaxSAT search state.

    The main state groups are:
    * problem/closure/lower_bound: input arrays and certified timing bounds;
    * horizon/domains/L/literal_rows: the current finite start-time encoding;
    * u and Boolean caches: reusable execution, conjunction, and group literals;
    * storage_pairs/chains/saturation: certificates enabling smaller encodings;
    * formula/model/objective_cost: WCNF input and the accepted RC2 result;
    * timing counters/trials: cumulative measurements across horizon attempts.

    The class mutates the supplied problem's horizon and time-window arrays.
    Use separate problem objects for solvers that must keep independent windows.
    """
    def __init__(self, problem, *, engine="g4", slack=16, cardinality="seqcounter", adapt=False, grouped=True):
        """Preprocess the instance and prepare state without building its clauses.

        Args:
            problem: SAMSPProblem-compatible object with consecutive activity IDs,
                zero-duration source/sink, integer timing data, renewable demands,
                and signed storage events. This method does not fully validate input.
            engine: PySAT SAT engine name passed to RC2; "g4" selects Glucose4.
            slack: Initial extra time above the certified lower bound. The input
                UB can cap the initial trial, but that trial is never below the
                lower bound. Future trials may exceed the original UB.
            cardinality: Attribute name in pysat.card.EncType, used for unit-weight
                capacity rows. An unknown name raises AttributeError via getattr.
                Weighted rows use a BDD pseudo-Boolean encoder instead.
            adapt: Forwarded to RC2's intrinsic at-most-one adaptation option.
            grouped: Whether unit-demand resource rows combine certified mutually
                exclusive intervals into chain literals. Certificates are still
                computed when this is false; only their use for grouping changes.

        Preprocessing adds structural timing information, forced renewable orders,
        resource-energy lags, and, when certified, storage FIFO lags. None of these
        steps supplies a feasible incumbent. Formula construction is deferred to
        build_model(). Timing fields initialized here do not measure this constructor.
        """
        # Store options independently: engine/adapt configure RC2, grouped controls
        # chain compression, and cardinality selects only the unit-weight row encoder.
        self.problem = problem
        self.engine = engine
        self.adapt = adapt
        self.grouped = grouped
        self.cardinality = getattr(CardType, cardinality)
        # Save the caller's horizon before build_model mutates problem.UB. It limits
        # only the first trial, not all later search horizons.
        self.original_horizon = problem.UB
        # First add nonnegative starts and completion-before-sink arcs. Then derive
        # forced orders for jobs that cannot share a renewable resource, followed by
        # energy-based lag strengthening. None signals a detected global conflict.
        self.closure = _closure(problem)
        if self.closure is not None:
            self.closure = disjunctive_resource_closure(problem, self.closure)
        if self.closure is not None:
            self.closure = strengthen_resource_lags(problem, self.closure)
        # A workload witness records release r, completion tail q, total energy E,
        # and capacity B, proving makespan ≥ r + q + ceil(E/B). Keep witnesses for
        # later saturation clauses; use the input LB only on the contradiction path.
        self.lower_bound, self.bound_witnesses = resource_lower_bound(problem, self.closure) if self.closure is not None else (problem.LB, [])
        self.horizon = max(self.lower_bound, min(self.original_horizon, self.lower_bound + slack))
        # These are placeholders until build_model initializes the reserved TRUE ID.
        # L maps (activity, threshold) to an order literal; u maps (activity, time)
        # to an execution proxy. model/cost describe the last accepted RC2 result.
        self.formula = WCNF()
        self.L = {}
        self.u = {}
        self.model = None
        self.objective_cost = None
        # Counters accumulate over horizon trials. native_seconds measures SAT-oracle
        # time, while optimization_seconds also includes RC2 overhead and rebuilds.
        self.encoding_seconds = 0.0
        self.optimization_seconds = 0.0
        self.native_seconds = 0.0
        self.calls = 0
        self.trials = []
        # Use interval storage only when every event belongs to a certified unit
        # pair. Otherwise keep the general signed-event representation for all zones.
        self.storage_pairs = self._certified_pairs() if self.closure is not None else None
        self.implied_storage_arcs = []
        # FIFO capacity implications can strengthen the timing matrix again. Recompute
        # the workload lower bound and initial horizon after those new consequences.
        if self.storage_pairs is not None:
            self.closure = self._storage_fifo_closure(self.closure)
            self.lower_bound, self.bound_witnesses = resource_lower_bound(problem, self.closure) if self.closure is not None else (problem.LB, [])
            self.horizon = max(self.lower_bound, min(self.original_horizon, self.lower_bound + slack))
        # A chain contains intervals that cannot overlap. Its OR can replace its
        # unit-weight sum in a capacity row. Renewable chains store activity IDs;
        # storage chains store indices into the zone's storage_pairs list.
        self.renewable_chains = {}
        self.storage_chains = {}
        if self.closure is not None:
            for r in range(problem.number_of_renewable_resources):
                jobs = [i for i in problem.activity_ids if problem.durations[i] and problem.renewable_requests[i][r]]
                # S[b] − S[a] ≥ d[a] proves a finishes before b starts. Merely having a
                # positive lag would not suffice when a lasts longer than that lag.
                self.renewable_chains[r] = self._chains(jobs, lambda a, b: self.closure[a, b] >= problem.durations[a])
            if self.storage_pairs is not None:
                for r, pairs in enumerate(self.storage_pairs):
                    # Stay a ends at completion of its release activity. Stay b starts at
                    # its acquisition activity, so compare those two events, not job starts.
                    self.storage_chains[r] = self._chains(list(range(len(pairs))),
                        lambda a, b: self.closure[pairs[a][1], pairs[b][0]] >= problem.durations[pairs[a][1]])
        # built is the build_model no-op guard. rc2 is non-None only during a native
        # optimization context. domains holds allowed starts; literal_rows makes
        # threshold lookup constant-time even when domains contain gaps.
        self.built = False
        self.rc2 = None
        self.domains = {}
        self.literal_rows = {}
        # On a capacity-one resource, a witness attaining the global lower bound
        # leaves no idle slot if the sink attains that bound. Retain only witnesses
        # whose unit-demand jobs reproduce the stated energy exactly.
        self.saturation = []
        for witness in self.bound_witnesses:
            if (witness['capacity'] == 1 and witness['lower_bound'] == self.lower_bound):
                r = witness['resource']
                jobs = [i for i in problem.activity_ids if problem.durations[i] > 0
                        and problem.renewable_requests[i][r] == 1
                        and self.closure[problem.source_id, i] >= witness['release']
                        and self.closure[i, problem.sink_id] - problem.durations[i] >= witness['tail']]
                # This equality checks the reconstructed subset, rather than assuming its
                # energy matches. Accepted jobs later require exact execution literals.
                if sum(problem.durations[i] for i in jobs) == witness['energy']:
                    self.saturation.append((witness, jobs))

    def _new(self):
        """Reserve and return the next unused positive Boolean variable ID.

        formula.nv is the largest reserved ID, including IDs allocated by PySAT's
        native encoders. Keeping this shared high-water mark prevents collisions.
        build_model reserves ID 1 for TRUE before this helper is used.
        """
        self.formula.nv += 1
        return self.formula.nv

    @staticmethod
    def _chains(items, precedes):
        """Partition items into chains whose neighboring items provably precede.

        ``precedes(a,b)`` is a caller-supplied temporal certificate, not an order
        inferred from IDs. For renewable jobs it certifies finish-before-start;
        for storage it can certify either interval separation or FIFO event order.
        Callers supply a transitive temporal relation. If both directions hold,
        increasing item IDs break the tie to avoid selecting both directions.

        The construction matches two copies of the item set as a bipartite graph.
        Each item gets at most one successor and one predecessor. Augmenting paths
        maximize the number of links; following those links produces the chains.
        Under the acyclic relation used here, this gives a minimum chain cover.

        Returns a list of lists, covering each input item once; [] for no items.
        Raises ValueError if reconstruction cannot cover all items because selected
        links contain a cycle. Items must be unique, hashable, and comparable for
        the ID tie-break. This method emits no Boolean variables or clauses.
        """
        # Make left-to-right candidate links between certified ordered items. Mutual
        # precedence can arise for equal-time events; orient that tie by item ID.
        edges = {a: [b for b in items if a != b and precedes(a, b)
                     and (not precedes(b, a) or a < b)] for a in items}
        matched = {}

        def match(a, seen):
            """Try to give left item a a right successor, rerouting earlier links.

            matched[b] stores b's current left predecessor. If b is occupied, recurse
            to move its predecessor elsewhere. ``seen`` contains right items visited
            during this attempt, preventing repeated exploration. Return True only
            when an augmenting path adds one link; mutate matched in that case.
            """
            for b in edges[a]:
                if b not in seen:
                    seen.add(b)
                    if b not in matched or match(matched[b], seen):
                        matched[b] = a
                        return True
            return False

        # Each search gets a fresh visited set. Existing links can be rerouted,
        # so this finds more links than choosing each item's first free neighbor.
        for a in items:
            match(a, set())
        # matched maps right item to left predecessor. Reverse it to follow chains.
        # Items absent as right keys have no predecessor and therefore start a chain.
        successors = {a: b for b, a in matched.items()}
        result = []
        for a in items:
            if a not in matched:
                chain = [a]
                while chain[-1] in successors:
                    chain.append(successors[chain[-1]])
                result.append(chain)
        # A selected cycle has no chain head, so reconstruction would omit its items.
        # Fail rather than silently drop resource contributions from a capacity row.
        if sum(map(len, result)) != len(items):
            raise ValueError('Certified non-overlap chains contained a cycle')
        return result

    def _storage_fifo_closure(self, closure):
        """Strengthen closure in place using capacity and certified FIFO chains.

        A pair (g,e) acquires at S[g] and releases at S[e] + d[e]. Each stay is
        nonnegative by _certified_pairs. If one chain's acquisitions AND releases
        are ordered, acquisition k + capacity cannot precede release k. Otherwise
        more than capacity stays from that chain alone would occupy the zone.
        This is a necessary constraint even when other chains share the zone.

        Add the lag S[get] − S[release] ≥ d[release] for each such pair. Record
        these arcs for explicit CNF encoding later. At most four rounds seek new
        consequences; this is a strengthening budget, not a full fixed-point claim.
        Return the mutated closure, or None if adding an implied arc conflicts.
        Zero/nonpositive capacities are skipped here and left to capacity encoding.
        """
        p = self.problem
        # Repeat because one new storage lag can certify additional event ordering.
        # Four is a bounded preprocessing effort; missing further lags only weakens it.
        for _ in range(4):
            changed = False
            for r, pairs in enumerate(self.storage_pairs):
                capacity = p.cumulative_capacities[r]
                if capacity <= 0:
                    continue
                # Acquisitions are ordered by their starts. Releases are ordered by their
                # COMPLETIONS: S[e_b] − S[e_a] ≥ d[e_a] − d[e_b] is the needed test.
                chains = self._chains(list(range(len(pairs))), lambda a, b:
                    closure[pairs[a][0], pairs[b][0]] >= 0 and
                    closure[pairs[a][1], pairs[b][1]] >= p.durations[pairs[a][1]] - p.durations[pairs[b][1]])
                for chain in chains:
                    # Compare positions k and k + capacity, not adjacent pairs unless capacity
                    # is one. The later acquisition must wait for the earlier release completion.
                    for before, after in zip(chain, chain[capacity:]):
                        release, get = pairs[before][1], pairs[after][0]
                        weight = p.durations[release]
                        # Only ask _add_arc for a strict improvement: it also returns False for a
                        # redundant arc, whereas False here is interpreted as a conflict.
                        if closure[release, get] < weight:
                            if not _add_arc(closure, release, get, weight):
                                return None
                            # Closure tightens windows; save the arc too so interior start choices
                            # in the Boolean model must obey this new storage implication.
                            self.implied_storage_arcs.append((release, get, weight))
                            changed = True
            if not changed:
                break
        return closure

    def _any(self, literals, exact=False):
        """Return a cached literal representing an OR, optionally exactly.

        Remove FALSE, fold TRUE, and reuse singleton inputs before allocating an
        auxiliary variable. With exact=False, each input implies the result, but
        the result may also be true with every input false. This direction is
        sufficient when the result is only counted in an upper capacity bound:
        spurious true values cannot conceal overload, and exact values are always
        available for a valid schedule. Chain grouping additionally relies on
        the actual intervals being mutually exclusive, so the OR of their exact
        occupancy values equals their unit-weight sum. One-way input proxies
        may have extra true values; an exact assignment remains available.

        With exact=True, add the reverse implication so the result equals the OR.
        This is required when its negation is counted as an idle slot. A later
        exact request upgrades the existing cached literal without changing its ID.
        """
        # Canonicalize commutative OR inputs for cache reuse. Constants use reserved
        # variable 1; no fresh variable is needed for an empty or singleton group.
        values = tuple(sorted(set(lit for lit in literals if lit != -1)))
        if not values:
            return -1
        if 1 in values:
            return 1
        if len(values) == 1:
            return values[0]
        if values not in self.group_literals:
            v = self._new()
            # Each clause is lit implies v. With only an upper capacity row, these
            # forward implications already force every actual occupied group to count.
            for lit in values:
                self._clause([-lit, v])
            self.group_literals[values] = v
        # [-v, *values] adds v implies at least one input. Keep a separate upgrade
        # set so repeated exact requests do not duplicate this reverse clause.
        if exact and values not in self.exact_groups:
            self._clause([-self.group_literals[values], *values])
            self.exact_groups.add(values)
        return self.group_literals[values]

    def _clause(self, literals):
        """Append a hard OR clause after inexpensive constant simplification.

        Literal 1 satisfies the clause, so discard that clause. Literal -1 never
        helps satisfy it, so remove -1. A two-literal [x,-x] is also always true.
        No general deduplication or long-clause tautology detection is attempted.
        If all literals disappear, retain [] to express a contradiction.

        Direct insertion avoids WCNF.append's copying and ID scanning. Every
        variable must therefore have been reserved already; this helper does not
        update formula.nv. It is for hard clauses only, never weighted soft clauses.
        """
        # A TRUE disjunct satisfies the whole clause; FALSE disjuncts can be erased.
        # An erased-to-empty clause must remain because it expresses impossibility.
        if 1 in literals:
            return
        values = [literal for literal in literals if literal != -1]
        if len(values) == 2 and values[0] == -values[1]:
            return
        # All IDs already come from _new or a registered native encoding.
        # Avoid WCNF.append's repeated copies and variable scans on hard rows.
        self.formula.hard.append(values)

    def get_L(self, i, t):
        """Return the literal for S[i] ≤ t, including constants outside the window.

        Before ES[i] the predicate is FALSE (-1); at or after LS[i] it is TRUE
        (1). Inside the window, use a dense lookup row that can alias several
        thresholds to one variable when the start domain has gaps. For example,
        with allowed starts [2,5,8], thresholds 2,3,4 all mean S[i] ≤ 2.

        Build the rows before calling this method for an interior threshold.
        This returns an integer literal, not its truth value in a solved model.
        """
        if t < self.problem.ES[i]:
            return -1
        if t >= self.problem.LS[i]:
            return 1
        return self.literal_rows[i][t - self.problem.ES[i]]

    def _and(self, a, b, exact=False):
        """Return a cached conjunction literal, with optional full equivalence.

        Fold constants, equal inputs, and complementary inputs first. Otherwise
        add [v,-a,-b], expressing (a ∧ b) implies v. Upper capacity bounds need
        only this direction: actual occupancy must be counted, and a feasible
        schedule can set v exactly. The default v is therefore a counting proxy,
        not necessarily a fully equivalent execution/stay indicator in every model.

        exact=True also adds [-v,a] and [-v,b], making v equivalent to a ∧ b.
        Busy-slot requirements need that reverse direction to prevent fictitious
        execution. Sort the pair to share a ∧ b with b ∧ a; later exact requests
        upgrade the cached proxy rather than allocate a second variable.
        """
        # FALSE and anything is FALSE; x and not-x is FALSE. TRUE and x is x;
        # x and x is also x. These cases avoid unnecessary auxiliary variables.
        if a == -1 or b == -1 or a == -b:
            return -1
        if a == 1 or a == b:
            return b
        if b == 1:
            return a
        # Sort signed literal IDs, not just variable IDs: negations are distinct
        # inputs, while swapping the two arguments must still reuse the same proxy.
        key = tuple(sorted((a, b)))
        if key not in self.conjunctions:
            v = self._new()
            # Capacity upper bounds need only (a AND b) => v. Extra true
            # values cannot hide an overload; a valid schedule can assign v
            # exactly. Busy-slot lower bounds also need the reverse direction.
            self._clause([v, -a, -b])
            self.conjunctions[key] = v
        # The reverse clauses v implies a and v implies b forbid a proxy from
        # claiming fictitious usage when later clauses require a busy interval.
        if exact and key not in self.exact_conjunctions:
            v = self.conjunctions[key]
            self._clause([-v, a])
            self._clause([-v, b])
            self.exact_conjunctions.add(key)
        return self.conjunctions[key]

    def _atmost(self, literals, bound, weights=None):
        """Encode a weighted upper bound after normalization and row reuse.

        The requested constraint is ∑[k] weights[k] · [literals[k]] ≤ bound,
        where [literal] is 1 when that signed literal is true and 0 otherwise.
        Weights default to 1. Callers supply matching lists and nonnegative integer
        weights; this helper does not validate list lengths or weight domains.

        Subtract TRUE contributions from bound, drop FALSE, merge repeated
        literals, and cancel complementary contributions using x + ¬x = 1.
        Cache the resulting row, since neighboring time slots often produce the
        same constraint. A negative bound is impossible; a bound at least the
        total remaining weight is automatic and needs no clauses.

        Unit coefficients use CardEnc with the selected cardinality encoding.
        Other coefficients use PBEnc.leq with a BDD (binary decision diagram).
        Both return CNF plus auxiliary IDs; extend the hard formula and reserve
        their highest ID so the next _new call cannot collide with an encoder.
        """
        if weights is None:
            weights = [1] * len(literals)
        # Normalize before choosing an encoder. For example, 2·TRUE + 3·x ≤ 4
        # becomes 3·x ≤ 2, and repeated x terms combine into one coefficient.
        coefficients = {}
        for literal, weight in zip(literals, weights):
            if literal == 1:
                bound -= weight
            elif literal != -1:
                coefficients[literal] = coefficients.get(literal, 0) + weight
        # Cancel a common positive amount from x and not-x. For example,
        # 3·x + 2·¬x ≤ 2 is equivalent to x ≤ 0 after subtracting constant 2.
        # Iterate a key snapshot because coefficients are updated during the pass.
        for literal in list(coefficients):
            if literal > 0 and -literal in coefficients:
                common = min(coefficients[literal], coefficients[-literal])
                bound -= common
                coefficients[literal] -= common
                coefficients[-literal] -= common
        # Remove zero coefficients introduced by cancellation. The normalized bound
        # and sorted signed coefficients uniquely identify this reusable capacity row.
        coefficients = {literal: weight for literal, weight in coefficients.items() if weight}
        key = (bound, tuple(sorted(coefficients.items())))
        if key in self.capacity_rows:
            return
        self.capacity_rows.add(key)
        # With nonnegative weights, even all-false cannot satisfy a negative bound.
        # Conversely, if all remaining terms can be true within bound, emit nothing.
        if bound < 0:
            self.formula.append([])
        elif bound < sum(coefficients.values()):
            # Choose cardinality only AFTER merging/cancellation: duplicate unit inputs
            # can become a weighted coefficient and then require the PB encoder.
            if all(weight == 1 for weight in coefficients.values()):
                encoded = CardEnc.atmost(list(coefficients), bound=bound, top_id=self.formula.nv, encoding=self.cardinality)
            else:
                encoded = PBEnc.leq(list(coefficients), weights=list(coefficients.values()), bound=bound,
                                    top_id=self.formula.nv, encoding=PBType.bdd)
            # top_id tells the encoder which IDs already exist. encoded.nv includes
            # its auxiliaries and must become the new high-water mark.
            self.formula.hard.extend(encoded.clauses)
            self.formula.nv = max(self.formula.nv, encoded.nv)

    def _certified_pairs(self):
        """Try to decompose every storage zone into nonnegative unit stays.

        Every positive unit event must match exactly one negative unit event in
        the same zone. A candidate (g,e) is certified if closure[g,e] + d[e] ≥ 0,
        proving S[g] ≤ S[e] + d[e] in every allowed schedule. Its release can never
        occur before its acquisition. Thus [acquired] − [released] equals the
        nonnegative stay predicate [acquired AND not released]. Summing all pairs
        reproduces every signed event exactly, so nonnegativity follows without
        separate lower rows. No physical sample/cuvette identity is asserted.

        A bipartite augmenting-path matching finds a complete pairing when one
        exists among these certified edges. Candidate sorting is a deterministic
        preference, not an optimal-weight assignment. Match constrained gets first.

        Return one list of (get, release) pairs per zone, including empty lists
        for empty zones. Return None if ANY zone has non-unit magnitudes, unequal
        event counts, or no complete certified matching. None selects the generic
        signed-storage encoding for ALL zones; it does not mean infeasibility.
        """
        result = []
        # Certification is per zone, but one failure returns None for the entire
        # instance. Empty zones yield empty pair lists and require no occupied stays.
        for r in range(self.problem.number_of_cumulative_resources):
            gets = [i for i in self.problem.activity_ids if self.problem.cumulative_requests[i][r] > 0]
            releases = [i for i in self.problem.activity_ids if self.problem.cumulative_requests[i][r] < 0]
            if any(abs(self.problem.cumulative_requests[i][r]) != 1 for i in gets + releases) or len(gets) != len(releases):
                return None
            # The edge test is a timing proof valid for every feasible schedule.
            # Prefer the smallest certified completion separation, then nearby IDs,
            # then release ID. The preference itself proves no physical item identity.
            edges = {g: sorted((e for e in releases if self.closure[g, e] + self.problem.durations[e] >= 0),
                               key=lambda e: (self.closure[g, e] + self.problem.durations[e], abs(e - g), e)) for g in gets}
            matched = {}

            def match(g, seen):
                """Match acquisition g to one certified release, rerouting earlier pairs.

                matched[e] stores the acquisition currently assigned to release e.
                ``seen`` tracks releases visited in this search. Reassigning a release's
                old acquisition can free that release for g. Return False if no augmenting
                path exists; the caller then falls back to signed-storage constraints.
                """
                for e in edges[g]:
                    if e not in seen:
                        seen.add(e)
                        if e not in matched or match(matched[e], seen):
                            matched[e] = g
                            return True
                return False

            # Try acquisitions with few certified releases first. A complete matching
            # uses every acquisition and release once, so no storage event is lost.
            if not all(match(g, set()) for g in sorted(gets, key=lambda g: len(edges[g]))):
                return None
            result.append([(g, e) for e, g in matched.items()])
        return result

    def build_model(self):
        """Build the bounded WCNF formula for the current horizon once.

        Stages: initialize constants/caches; propagate the trial horizon; derive
        finite domains; create monotone order literals; encode lag implications;
        enforce renewable rows, saturation consequences, and storage rows; then
        attach makespan soft clauses. Comments below explain each construction.

        An impossible preprocessing result or empty domain adds [] rather than
        raising a scheduling-infeasibility exception. RC2 then detects hard UNSAT.
        The method returns None and marks built=True even for that contradiction.
        Repeated calls are no-ops until solve sets built=False after horizon growth.

        Mutates problem.UB and ES/LS/EC/LC for the trial. Formula literals and
        caches are rebuilt; variable IDs from an earlier trial must not be reused.
        encoding_seconds accumulates build time, including trial propagation but
        excluding constructor preprocessing. Domain arrays describe this trial only.
        """
        # The runner may call build_model repeatedly. Reuse the current formula
        # until solve explicitly invalidates it after increasing the horizon.
        if self.built:
            return
        started = time.monotonic()
        # 1. Initialize the bounded formula and reserve the TRUE variable.
        # Clause [1] makes integer literal -1 a usable FALSE constant.
        self.formula = WCNF()
        self.formula.nv = 1
        self.formula.append([1])
        # Every auxiliary cache is tied to this formula's IDs. Reset all caches
        # on rebuild so a previous horizon's literals cannot leak into new clauses.
        self.L, self.u, self.conjunctions, self.capacity_rows, self.literal_rows = {}, {}, {}, set(), {}
        self.exact_conjunctions = set()
        self.group_literals = {}
        self.exact_groups = set()
        # 2. Propagate S[sink] ≤ horizon in a COPY of the globally valid closure.
        # Orders inferred under this horizon must not contaminate a later, larger
        # trial. None here can mean only this horizon is too small.
        trial_closure = None if self.closure is None else disjunctive_resource_closure(self.problem, self.closure, self.horizon)
        if trial_closure is None:
            self.formula.append([])
        else:
            p = self.problem
            # Refresh the problem's public horizon/windows, then replace its windows
            # with the stronger trial-specific closure bounds below.
            p.set_search_horizon(self.horizon)
            # 3. Derive optional start-time residues for a fully saturated unary machine.
            # If all witness jobs last step time units and horizon equals the workload
            # bound, they tile its window without gaps; starts equal release modulo step.
            # A job present in several witnesses must satisfy every collected residue.
            residues = {}
            if self.horizon == self.lower_bound:
                for witness, jobs in self.saturation:
                    durations = {p.durations[i] for i in jobs}
                    if len(durations) == 1:
                        step = next(iter(durations))
                        for i in jobs:
                            residues.setdefault(i, []).append((witness['release'], step))
            # A source-to-job lower lag gives earliest start. A job-to-sink lower lag
            # and sink ≤ horizon give latest start. Structural arcs make these finite.
            for i in p.activity_ids:
                p.ES[i] = int(trial_closure[p.source_id, i])
                p.LS[i] = self.horizon - int(trial_closure[i, p.sink_id])
                if i == p.source_id:
                    p.ES[i] = p.LS[i] = 0
                if i == p.sink_id:
                    p.ES[i] = max(p.ES[i], self.lower_bound)
                # Retain integer starts inside both temporal bounds and any residue classes.
                # Without residue certificates this is the full contiguous start window.
                self.domains[i] = [t for t in range(p.ES[i], p.LS[i] + 1)
                                   if all((t - release) % step == 0 for release, step in residues.get(i, []))]
                # An empty domain already proves this trial impossible. Add [] but keep
                # one placeholder threshold so the remaining construction remains indexable;
                # the placeholder cannot make the contradictory formula satisfiable.
                if not self.domains[i]:
                    self.formula.append([])
                    self.domains[i] = [p.LS[i]]
                # Tighten public endpoints to the filtered domain. Completion endpoints are
                # start endpoints plus duration, even if some interior starts are excluded.
                p.ES[i], p.LS[i] = self.domains[i][0], self.domains[i][-1]
                p.EC[i] = p.ES[i] + p.durations[i]
                p.LC[i] = p.LS[i] + p.durations[i]
                # Allocate one order variable per allowed start except the last. At the
                # last allowed start the job has certainly started, so reuse TRUE (ID 1).
                # A fixed-start job therefore needs no decision variable of its own.
                for t in self.domains[i][:-1]:
                    self.L[i, t] = self._new()
                self.L[i, self.domains[i][-1]] = 1
                if p.ES[i] > p.LS[i]:
                    self.formula.append([])
            # 4. Expand sparse domains into fast threshold lookup rows. bisect_right
            # selects the largest allowed start at most t. If starts are [2,5,8],
            # thresholds 2 through 4 share the variable for 2. No new choice is added.
            for i in p.activity_ids:
                domain = self.domains[i]
                self.literal_rows[i] = [self.L[i, domain[bisect_right(domain, t) - 1]]
                                        for t in range(p.ES[i], p.LS[i] + 1)]
                # Expose the same aliases through L for callers decoding all integer
                # thresholds. literal_rows remains the fast path used by get_L.
                for t in range(p.ES[i], p.LS[i] + 1):
                    self.L[i, t] = self.get_L(i, t)
            for i in p.activity_ids:
                domain = self.domains[i]
                # 5. Enforce monotonicity: started by earlier implies started by later.
                # The final domain threshold is already TRUE, so its incoming implication
                # is automatic. A false-prefix/true-suffix row encodes one integer start.
                for earlier, later in zip(domain[:-2], domain[1:-1]):
                    self._clause([-self.get_L(i, earlier), self.get_L(i, later)])
            self._clause([self.get_L(p.source_id, 0)])
            # 6. Encode original lags, completion-before-sink arcs, and implied FIFO
            # arcs. Stronger closure information also acts through windows/certificates;
            # the full all-pairs closure is not materialized as Boolean lag clauses.
            arcs = [(a['from'], a['to'], a['lag']) for a in p.lags]
            arcs += [(i, p.sink_id, p.durations[i]) for i in p.activity_ids if i != p.sink_id]
            arcs += self.implied_storage_arcs
            # S[b] − S[a] ≥ weight is encoded backwards from each possible start
            # threshold of b: S[b] ≤ t implies S[a] ≤ t − weight. This works for
            # negative weights too. Include b's final threshold: it may force a bound.
            for a, b, weight in arcs:
                for t in self.domains[b]:
                    self._clause([-self.get_L(b, t), self.get_L(a, t - weight)])
            # 7. Encode renewable usage for integer slots 0 through horizon − 1.
            # Jobs used in busy-slot constraints need fully equivalent execution
            # literals. Other jobs need only proxies forced true during real execution.
            exact_usage = {i for _, jobs in self.saturation for i in jobs}
            for r, capacity in enumerate(p.renewable_capacities):
                jobs = [i for i in p.activity_ids if p.durations[i] and p.renewable_requests[i][r]]
                for t in range(self.horizon):
                    literals, weights = [], []
                    for i in jobs:
                        # Outside this possible-execution interval the job cannot use the slot.
                        # Inside it, started by t AND not finished by t means running at t.
                        # For d=2 and S=3, usage is true at t=3,4 and false at t=5.
                        if p.ES[i] <= t < p.LC[i]:
                            literal = self._and(self.get_L(i, t), -self.get_L(i, t - p.durations[i]),
                                                exact=i in exact_usage)
                            self.u[i, t] = literal
                            literals.append(literal)
                            weights.append(p.renewable_requests[i][r])
                    # Each certified chain has at most one actually executing job, so
                    # the OR of exact usage values equals its unit demand. Proxy values
                    # can overcount, but real chain occupancy is always counted and a
                    # valid schedule can choose exact proxies. Count groups instead of
                    # jobs. Weighted demands need the PB row: an OR loses their weights.
                    if self.grouped and all(weight == 1 for weight in weights):
                        literals = [self._any([self.u.get((i, t), -1) for i in chain])
                                    for chain in self.renewable_chains[r]]
                        self._atmost(literals, capacity)
                    else:
                        self._atmost(literals, capacity, weights)
            # 8. Strengthen tight unary-resource witnesses. If sink ≤ lower_bound,
            # the certified lower bound is attained and every witness-window slot
            # must be occupied by one witness job. At larger makespans this first
            # requirement is disabled by its negative sink guard.
            # Equality in the energy bound requires uninterrupted occupation by
            # the witness jobs. The implication is guarded by the sink bound.
            for witness, jobs in self.saturation:
                for t in range(witness['release'], self.lower_bound - witness['tail']):
                    self._clause([-self.get_L(p.sink_id, self.lower_bound)] +
                                 [self.u.get((i, t), -1) for i in jobs])
                # The witness's jobs occupy exactly energy distinct unit slots.
                # Counting the few remaining idle slots exposes tight workload
                # bounds to SAT propagation across the entire time interval.
                # Over the full trial window, all witness work occupies exactly energy
                # slots on the unary machine. The remaining slots are idle. Limit their
                # count when the budget is small (0..16); 16 is an encoding-effort cutoff,
                # not a scheduling restriction. Exact usage/OR prevents fictitious busy slots.
                budget = self.horizon - witness['release'] - witness['tail'] - witness['energy']
                if 0 <= budget <= 16:
                    busy = [self._any([self.u.get((i, t), -1) for i in jobs], exact=True)
                            for t in range(witness['release'], self.horizon - witness['tail'])]
                    self._atmost([-lit for lit in busy], budget)
            # 9. Encode storage at boundaries 0 through horizon INCLUSIVE. Positive
            # events occur at start; negative events occur at completion. The same
            # timestamp handles both together, without an artificial event sequence.
            for r, capacity in enumerate(p.cumulative_capacities):
                for t in range(self.horizon + 1):
                    # Certified pairs make every stay nonnegative. Their sum exactly equals
                    # signed occupancy, so only an upper capacity row is needed. The release
                    # test uses t − d[e] because release happens when e COMPLETES.
                    if self.storage_pairs is not None:
                        literals = [self._and(self.get_L(g, t), -self.get_L(e, t - p.durations[e]))
                                    for g, e in self.storage_pairs[r]]
                        # Certified non-overlap chains permit the same unit OR compression
                        # as renewable chains, but here each item is an entire storage stay.
                        if self.grouped:
                            literals = [self._any([literals[i] for i in chain]) for chain in self.storage_chains[r]]
                        self._atmost(literals, capacity)
                    else:
                        # Generic signed storage requires BOTH bounds. Let A be total released
                        # amount and Q the nonnegative weighted sum built below. A release term
                        # −w·L[e,t−d[e]] becomes w·¬L[e,t−d[e]] − w, hence occupancy O = Q − A.
                        literals, weights, total_release = [], [], 0
                        for i in p.activity_ids:
                            delta = p.cumulative_requests[i][r]
                            if delta > 0:
                                literals.append(self.get_L(i, t))
                                weights.append(delta)
                            elif delta < 0:
                                literals.append(-self.get_L(i, t - p.durations[i]))
                                weights.append(-delta)
                                total_release -= delta
                        # O ≤ capacity becomes Q ≤ capacity + A. O ≥ 0 becomes Q ≥ A;
                        # with W = sum(weights), complementing every literal gives
                        # W − Q ≤ W − A. This second row prevents releases before acquisitions.
                        self._atmost(literals, capacity + total_release, weights)
                        self._atmost([-lit for lit in literals], sum(weights) - total_release, weights)
            # 10. Make the certified lower bound explicit, then attach the objective.
            # For sink start M, soft thresholds lower_bound through M − 1 are false;
            # exactly M − lower_bound unit penalties result. If horizon equals the
            # lower bound, there are no soft clauses and any hard model has cost zero.
            self._clause([-self.get_L(p.sink_id, self.lower_bound - 1)])
            for t in range(self.lower_bound, self.horizon):
                self.formula.append([self.get_L(p.sink_id, t)], weight=1)
            # Zero-duration jobs may have entirely unconstrained order variables.
            # Register the highest ID so the native model includes those IDs too.
            self.formula.append([self.formula.nv, -self.formula.nv])
        # Mark completion even if [] proved the trial impossible. solve needs a
        # native UNSAT result to decide whether to expand this trial's horizon.
        self.encoding_seconds += time.monotonic() - started
        self.built = True

    def solve(self, time_limit=None):
        """Optimize across horizon trials under one optional wall-clock budget.

        Call build_model() first. ``time_limit`` is seconds from entry to this
        method; None disables the deadline, and a nonpositive value returns UNKNOWN.
        The initial build and constructor happen outside this budget. Rebuilds
        occur inside it, but are synchronous and are checked again after returning;
        the method cannot interrupt their Python work at the exact deadline.

        RC2 solves each trial to optimality. A hard-UNSAT trial expands the horizon
        unless closure is None, which is a preprocessing proof of global conflict.
        Return OPTIMAL for an accepted optimum, UNKNOWN for interruption, or
        UNSATISFIABLE for that global preprocessing conflict. General resource
        infeasibility without such a conflict can keep expanding until timeout.

        Each completed _compute call adds a horizon/status record. Timing and
        oracle-call counters accumulate across calls; this is not incremental
        reuse of an RC2 object between horizons. Exceptions propagate after timing
        accounting. Deadline checks that return before _compute add no trial record.
        """
        # Use one absolute deadline, so each horizon attempt receives only the
        # remaining budget rather than a fresh time_limit.
        started = time.monotonic()
        deadline = None if time_limit is None else started + time_limit
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    return SOLVER_STATUS.UNKNOWN
                # The initial formula must already exist. _compute creates a fresh native
                # optimizer, and the recorded status describes this exact horizon.
                status = self._compute(deadline)
                self.trials.append(dict(horizon=self.horizon, status=status.name))
                # A global preprocessing conflict does not disappear at a larger horizon.
                # Otherwise only trial UNSAT triggers growth; OPTIMAL/UNKNOWN return as-is.
                if status != SOLVER_STATUS.UNSATISFIABLE or self.closure is None:
                    return status
                # Increase at least one slot and roughly double slack above the bound.
                # For lower_bound=10 and horizons 10,12,16, the next values are 12,16,24.
                # original_horizon is deliberately no longer a cap on this feasibility search.
                self.horizon = max(self.horizon + 1, self.lower_bound + 2 * (self.horizon - self.lower_bound + 1))
                # Rebuild order domains and all clauses for the larger horizon. Even if
                # the rebuild consumes the remaining budget, the next loop check returns
                # UNKNOWN rather than granting another full solve budget.
                self.built = False
                self.build_model()
        finally:
            self.optimization_seconds += time.monotonic() - started

    def _compute(self, deadline):
        """Run one RC2 optimization on the already built bounded formula.

        ``deadline`` is an absolute time.monotonic() timestamp, or None. Construct
        a fresh RC2 context, count its SAT-oracle calls, and schedule a cooperative
        interruption timer using the remaining budget. g3/g4 enable RC2's solver
        incrementality option inside this one RC2 run; horizon changes still rebuild.

        Accept a model only if neither the timer flag nor RC2 reports interruption.
        Save its objective cost and verify cost = decoded makespan − lower_bound.
        OPTIMAL here is for the current horizon; solve explains why that suffices
        globally. A None model without interruption means this hard formula is
        UNSATISFIABLE, not necessarily that the unbounded scheduling instance is.

        Native interruption support depends on the engine. If unsupported, the
        timer flag still rejects a late result, but cannot stop the blocked native
        call; an external process guard is needed for a hard runtime limit.
        Cleanup cancels/joins the timer, records native SAT time, and clears the
        live RC2 reference. The context manager releases native solver resources.
        """
        # Thread-safe local flag distinguishes deadline interruption from hard
        # UNSAT even if the native solver returns no model in either situation.
        interrupted = Event()
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return SOLVER_STATUS.UNKNOWN
        # The context owns the native solver. RC2 may make many SAT calls while
        # relaxing unsatisfiable soft-clause cores; incr applies within this context.
        with RC2(self.formula, solver=self.engine, adapt=self.adapt, incr=self.engine in {'g3', 'g4'}) as rc2:
            self.rc2 = rc2
            # RC2 construction also consumes budget, so recompute the remaining
            # seconds before starting its native optimization timer.
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self.rc2 = None
                return SOLVER_STATUS.UNKNOWN
            # Wrap RC2's internal oracle hook to measure SAT calls, not just horizon
            # attempts. Preserve the original bound method for transparent forwarding.
            original = rc2._call_oracle

            def counted(*args, **kwargs):
                """Count one RC2 SAT-oracle invocation and forward every argument unchanged.

                RC2 calls this wrapper during optimization; a call is not necessarily an
                entire MaxSAT solve. ``original`` is the bound oracle method saved before
                replacement, so forwarding does not recurse into this wrapper.
                """
                self.calls += 1
                return original(*args, **kwargs)

            rc2._call_oracle = counted

            def interrupt():
                """Mark the deadline event and request cooperative RC2 interruption.

                Timer runs this callback on another thread. Set the local Event first so
                even an engine without interrupt support cannot have a late result accepted.
                NotImplementedError is ignored only for the interruption request; other
                exceptions are not handled here. This callback does not manufacture UNSAT.
                """
                interrupted.set()
                try:
                    rc2.interrupt()
                except NotImplementedError:
                    # Some native engines require the runner's process guard.
                    # Never accept a result after this deadline, even if the
                    # native call cannot cooperate with interruption.
                    pass

            # Only bounded calls need a timer thread. Native interrupt is cooperative;
            # timer cancellation and joining below prevent callbacks leaking to later trials.
            timer = None if remaining is None else Timer(remaining, interrupt)
            if timer is not None:
                timer.start()
            try:
                # Normal completion returns an optimum Boolean model, or None for hard
                # UNSAT. Check interruption BEFORE interpreting None or accepting an optimum.
                model = rc2.compute(expect_interrupt=True)
                if interrupted.is_set() or rc2.interrupted:
                    self.model = None
                    return SOLVER_STATUS.UNKNOWN
                if model is None:
                    self.model = None
                    return SOLVER_STATUS.UNSATISFIABLE
                self.model = model
                # RC2 cost counts violated soft weights. Decode the sink independently
                # and cross-check the objective construction before reporting OPTIMAL.
                # This check does not replace independent validation of the full schedule.
                self.objective_cost = rc2.cost
                if self.get_makespan() - self.lower_bound != rc2.cost:
                    raise ValueError('Decoded makespan disagrees with MaxSAT objective')
                return SOLVER_STATUS.OPTIMAL
            finally:
                # Always clean up, including on decoding exceptions. join waits for an
                # already running callback, so no timer accesses RC2 after context teardown.
                if timer is not None:
                    timer.cancel()
                    timer.join()
                # SAT time excludes Python encoding and most RC2 orchestration overhead.
                # Clear the live handle before the context manager destroys the native solver.
                self.native_seconds += rc2.oracle.time_accum()
                self.rc2 = None

    def get_last_feasible_model(self):
        """Return the stored signed Boolean assignment, or None if none is retained.

        This returns the list itself, not a copy or a decoded schedule. In PySAT's
        ordered model convention, entry abs(lit) − 1 gives that variable's signed
        assignment. For positive order literals, a positive entry means true.
        Variable 1 is included, so fixed TRUE thresholds decode the same way.

        _compute clears this field for native UNSAT or interrupted results. Early
        deadline exits do not clear an older model; callers should inspect the
        current solve status and avoid treating retained state as a new result.
        """
        return self.model

    def get_makespan(self):
        """Decode the sink's first true threshold as the integer makespan.

        Return None if no model is stored. Monotonicity means all thresholds before
        the sink start are false and all thresholds at/after it are true. get_L
        handles fixed boundaries and sparse-domain aliases, so scanning the full
        integer window remains valid. The final threshold uses forced variable 1.

        Assumes the stored model and current problem windows belong to the same
        built formula. A malformed model with no true threshold raises StopIteration;
        the solver does not silently substitute the horizon.
        """
        if self.model is None:
            return None
        p = self.problem
        # Order literals use positive variable IDs (or TRUE=1). Under PySAT's
        # ordered assignment convention, subtract one to index the model list.
        return next(t for t in range(p.ES[p.sink_id], p.LS[p.sink_id] + 1)
                    if self.model[self.get_L(p.sink_id, t) - 1] > 0)

    def get_statistics(self):
        """Return current formula sizes, last cost, certificates, and cumulative work.

        variables includes reserved/auxiliary IDs. clauses and hard_clauses are
        the same hard-clause count; soft_clauses is separate. objective_cost is the
        last accepted RC2 cost (None before success); lower_bound + that cost gives
        the associated makespan. objective_upper_bound is the TRIAL HORIZON, not
        necessarily a feasible incumbent or a proven upper bound on the optimum.

        encoding_seconds counts build_model work; optimization_seconds counts
        solve wall time, including rebuilds; total_solving_time counts native SAT
        oracle time. These overlap and must not simply be added. Constructor
        preprocessing is outside them. number_of_calls counts RC2 oracle calls.
        Sizes describe the latest trial; counters include earlier trials as well.

        certified_storage says all zones have a certified unit pairing, including
        vacuous success when there are no zones. bound_witnesses holds resource,
        release, tail, energy, capacity, and lower_bound evidence. horizon_trials
        holds completed trial statuses. The returned dictionary shares those lists
        with the solver, so callers should treat their contents as read-only.
        """
        return dict(variables=self.formula.nv, clauses=len(self.formula.hard), hard_clauses=len(self.formula.hard),
                    soft_clauses=len(self.formula.soft), objective_cost=self.objective_cost,
                    objective_lower_bound=self.lower_bound, objective_upper_bound=self.horizon,
                    encoding_seconds=self.encoding_seconds, optimization_seconds=self.optimization_seconds,
                    total_solving_time=self.native_seconds, number_of_calls=self.calls,
                    certified_storage=self.storage_pairs is not None, bound_witnesses=self.bound_witnesses,
                    horizon_trials=self.trials, optimization='maxsat', engine=self.engine)
