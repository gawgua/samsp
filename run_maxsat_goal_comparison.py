"""Fresh, frozen-source MaxSAT/CP comparisons for Set I or Set II."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).absolute().parent
CP_ENGINE = Path('/opt/ibm/ILOG/CPLEX_Studio222/cpoptimizer/bin/x86-64_linux/cpoptimizer')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(folder, manifest, results):
    pairs = []
    table = ['| Instance | Repeat | N | CP status | MaxSAT status | CP makespan | MaxSAT makespan | CP wall s | MaxSAT wall s | CP/MaxSAT | Outcome |',
             '| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    indexed = {(row['instance'], row['repeat'], row['solver']): row for row in results}
    for repeat in range(manifest['repeats']):
        for item in manifest['instances']:
            a = indexed.get((item['path'], repeat, 'cp'))
            b = indexed.get((item['path'], repeat, 'maxsat_optimized'))
            if a is None or b is None:
                continue
            comparable = all(row.get('status') == 'OPTIMAL' and row.get('valid') is True for row in (a, b))
            agreement = comparable and a['makespan'] == b['makespan']
            ratio = a['wall_seconds'] / b['wall_seconds'] if agreement else None
            outcome = ('maxsat_win' if ratio is not None and ratio > 1 else 'cp_win' if ratio is not None and ratio < 1
                       else 'tie' if ratio == 1 else 'optimum_mismatch' if comparable else 'unresolved')
            pairs.append(dict(instance=item['name'], repeat=repeat, outcome=outcome,
                              cp_wall_seconds=a['wall_seconds'], maxsat_wall_seconds=b['wall_seconds'], ratio=ratio))
            ratio_text = f'{ratio:.3f}' if ratio is not None else '—'
            table.append(f"| {item['name']} | {repeat + 1} | {item['activities']} | {a['status']} | {b['status']} | "
                         f"{a.get('makespan', '—')} | {b.get('makespan', '—')} | {a['wall_seconds']:.3f} | "
                         f"{b['wall_seconds']:.3f} | {ratio_text} | {outcome} |")
    outcomes = {name: sum(p['outcome'] == name for p in pairs)
                for name in ('maxsat_win', 'cp_win', 'tie', 'optimum_mismatch', 'unresolved')}
    summary = dict(completed_runs=len(results), expected_runs=2 * len(manifest['instances']) * manifest['repeats'], outcomes=outcomes, pairs=pairs,
                   timing='Cold worker process wall time, including imports, loading, all bound calculations, construction, proof, validation and cleanup.',
                   historical_answers_consumed=False,
                   storage='CP paired intervals; optimized MaxSAT certifies each interval from timing, otherwise encodes both aggregate bounds.')
    (folder / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    (folder / 'comparison.md').write_text('# Standalone optimized MaxSAT versus CP\n\n' + summary['timing'] + '\n\n'
                                        + str(outcomes) + '\n\n' + '\n'.join(table) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark_results/maxsat_goal')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dataset', choices=('set_i', 'set_ii'), default='set_i')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--engine', default='g4')
    parser.add_argument('--slack', type=int, default=0)
    parser.add_argument('--cardinality', default='seqcounter')
    parser.add_argument('--adapt', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.resume:
        folder = args.resume.absolute()
        manifest = json.loads((folder / 'manifest.json').read_text())
    else:
        folder = args.output.absolute() / ('paired_' + time.strftime('%Y%m%d-%H%M%S'))
        folder.mkdir(parents=True, exist_ok=False)
        dataset_dir = ROOT / 'dataset/data' / args.dataset
        files = list(dataset_dir.glob('*.json'))
        expected_files = {'set_i': 27, 'set_ii': 24}[args.dataset]
        if len(files) != expected_files:
            raise ValueError(f'Expected {expected_files} {args.dataset} files, found {len(files)}')
        items = [dict(path=str(f), name=f.name, activities=len(json.loads(f.read_text())['activities'])) for f in files]
        items.sort(key=lambda x: (x['activities'], x['name']))
        source_hashes = {}
        sources = [ROOT / 'run_dataset.py', Path(__file__).absolute()]
        sources += list((ROOT / 'solver').glob('*.py')) + list((ROOT / 'dataset').glob('*.py'))
        sources += [ROOT / 'tests/test_maxsat_optimized.py']
        for source in sources:
            relative = source.relative_to(ROOT)
            target = folder / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            source_hashes[str(relative)] = digest(source)
        manifest = dict(dataset=args.dataset, dataset_dir=str(dataset_dir),
                        instances=items, repeats=args.repeats, time_limit=120, memory_limit_mb=4096,
                        cp_workers=1, cp_executable=str(CP_ENGINE), cp_sha256=digest(CP_ENGINE),
                        python_executable=sys.executable, python=sys.version, platform=platform.platform(),
                        load_at_start=os.getloadavg(), poll_interval=0.02, source_sha256=source_hashes,
                        input_sha256={str(f): digest(f) for f in files},
                        packages={p: importlib.metadata.version(p) for p in ('python-sat', 'networkx', 'numpy', 'docplex')},
                        engine=args.engine, slack=args.slack, cardinality=args.cardinality, adapt=args.adapt,
                        historical_answers_consumed=False)
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    frozen = folder / 'sources'
    sys.path.insert(0, str(frozen))
    spec = importlib.util.spec_from_file_location('frozen_goal_runner', frozen / 'run_dataset.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    results_file = folder / 'results.jsonl'
    results = [json.loads(line) for line in results_file.read_text().splitlines()] if results_file.exists() else []
    completed = {(r['instance'], r['repeat'], r['solver']) for r in results}
    settings = SimpleNamespace(dataset_dir=Path(manifest.get('dataset_dir', str(ROOT / 'dataset/data/set_i'))), license=None, savilerow_jar=None,
                               cp_executable=CP_ENGINE, cp_workers=1, time_limit=120, memory_limit_mb=4096,
                               poll_interval=manifest['poll_interval'], maxsat_engine=manifest['engine'],
                               maxsat_slack=manifest['slack'], maxsat_cardinality=manifest['cardinality'],
                               maxsat_adapt=manifest['adapt'])
    print(f'Results: {folder}', flush=True)
    with results_file.open('a') as stream:
        for repeat in range(manifest['repeats']):
            subfolder = folder / f'repeat_{repeat + 1}'
            subfolder.mkdir(exist_ok=True)
            for index, item in enumerate(manifest['instances']):
                order = ('cp', 'maxsat_optimized') if (index + repeat) % 2 == 0 else ('maxsat_optimized', 'cp')
                for backend in order:
                    key = (item['path'], repeat, backend)
                    if key in completed:
                        continue
                    path = Path(item['path'])
                    if digest(path) != manifest['input_sha256'][str(path)]:
                        raise ValueError(f'Input changed: {path}')
                    if any(digest(frozen / p) != h for p, h in manifest['source_sha256'].items()):
                        raise ValueError('Frozen source changed')
                    result = runner.run_one(settings, path, backend, subfolder)
                    result.update(repeat=repeat, input_sha256=manifest['input_sha256'][str(path)],
                                  engine_settings=dict(engine=manifest['engine'], slack=manifest['slack'],
                                                       cardinality=manifest['cardinality'], adapt=manifest['adapt'])
                                  if backend == 'maxsat_optimized' else dict(engine='CP Optimizer', Workers=1))
                    target = subfolder / (path.stem + '__' + backend + '.json')
                    runner.write_json(target, result)
                    if result.get('schedule') is not None:
                        runner.write_json(target.with_suffix('.schedule.json'), dict(schedule=result['schedule'],
                                                                                   valid=result.get('valid'), instance=str(path)))
                    results.append(result)
                    stream.write(json.dumps(result) + '\n')
                    stream.flush()
                    report(folder, manifest, results)
                    total_runs = 2 * len(manifest['instances']) * manifest['repeats']
                    print(f"[{len(results)}/{total_runs}] {path.name} {backend}: {result['status']} "
                          f"valid={result.get('valid')} makespan={result.get('makespan')} wall={result['wall_seconds']:.3f}s", flush=True)
    print(json.dumps(report(folder, manifest, results)['outcomes']), flush=True)
    print((folder / 'comparison.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
