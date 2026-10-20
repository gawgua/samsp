# Verified 3-repeat comparison

Cold worker wall time includes imports, input loading, all bounds, formula construction, proof, validation, and cleanup. Each pair is sequential; backend order alternates. CP uses one worker. Both use the same Python environment, 120-second total wall limit, and 4096 MB combined RSS guard. No CP result is used by MaxSAT.

Independent raw-JSON validation: 162 schedules pass. All paired optima agree. MaxSAT wins 27/27 instance medians and 81/81 individual pairs. CP wins 0 pairs; zero unresolved or invalid runs.

Total measured wall: CP 221.447 s; MaxSAT 111.426 s; ratio 1.987×. Penalized totals are identical because no run is unresolved.

| Instance | N | Optimum | CP median s | MaxSAT median s | CP/MaxSAT | MaxSAT pair wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| {15}.json | 27 | 53 | 0.784 | 0.524 | 1.496× | 3/3 |
| {35}.json | 37 | 64 | 0.780 | 0.513 | 1.520× | 3/3 |
| {45}.json | 37 | 58 | 0.783 | 0.512 | 1.529× | 3/3 |
| {110}.json | 52 | 85 | 0.833 | 0.549 | 1.517× | 3/3 |
| {15-35}.json | 62 | 74 | 1.254 | 0.638 | 1.966× | 3/3 |
| {15-45}.json | 62 | 76 | 1.085 | 0.640 | 1.695× | 3/3 |
| {25}.json | 62 | 91 | 0.817 | 0.583 | 1.401× | 3/3 |
| {310}.json | 72 | 96 | 0.959 | 0.583 | 1.645× | 3/3 |
| {35-45}.json | 72 | 82 | 1.186 | 0.621 | 1.910× | 3/3 |
| {410}.json | 72 | 90 | 0.934 | 0.583 | 1.602× | 3/3 |
| {15-25}.json | 87 | 101 | 0.821 | 0.621 | 1.322× | 3/3 |
| {25-35}.json | 97 | 103 | 1.421 | 0.746 | 1.905× | 3/3 |
| {25-45}.json | 97 | 101 | 2.530 | 0.643 | 3.935× | 3/3 |
| {120}.json | 102 | 155 | 0.842 | 0.572 | 1.472× | 3/3 |
| {110-310}.json | 122 | 144 | 2.153 | 1.150 | 1.872× | 3/3 |
| {110-410}.json | 122 | 146 | 2.049 | 1.210 | 1.693× | 3/3 |
| {210}.json | 122 | 161 | 0.954 | 0.690 | 1.383× | 3/3 |
| {310-410}.json | 142 | 162 | 1.712 | 1.098 | 1.559× | 3/3 |
| {320}.json | 142 | 176 | 1.527 | 0.725 | 2.106× | 3/3 |
| {420}.json | 142 | 170 | 1.584 | 0.725 | 2.185× | 3/3 |
| t1x10_t2x10_s0.json | 172 | 191 | 1.108 | 0.955 | 1.160× | 3/3 |
| t1x10_t2x10_s1.json | 172 | 191 | 1.099 | 1.065 | 1.032× | 3/3 |
| t1x10_t2x10_s2.json | 172 | 191 | 1.083 | 0.994 | 1.090× | 3/3 |
| {110-210}.json | 172 | 191 | 1.064 | 0.992 | 1.073× | 3/3 |
| {210-310}.json | 192 | 200 | 13.615 | 7.338 | 1.855× | 3/3 |
| {210-410}.json | 192 | 200 | 28.643 | 10.159 | 2.819× | 3/3 |
| {220}.json | 242 | 311 | 1.951 | 1.346 | 1.449× | 3/3 |

Phase times, RSS, variable/clause counts, objective costs, and runtime ranges are in `median_comparison.csv`. Encoding time accumulates across trial horizons. The initial build column covers the first formula. Optimization wall includes RC2 initialization and any later formula rebuilds; native time covers oracle solving. Those columns overlap and must not be summed. Bound preparation and startup are included in total wall.

These are observed results for this dataset, machine, CP configuration, and solver versions. They do not establish that MaxSAT is universally faster than CP or eliminate runtime variation.
