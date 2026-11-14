"""Outcome admission shared by one-shot and adaptive adapter consolidation."""
from pathlib import Path
import shutil

from .behavioral import compare
from .behavioral_job import load_resources, measure_selection, prepare_resources, resource_identity, evidence_files, verify_evidence


def sequential_report(report, ordinal):
    alpha = .05 / (ordinal * (ordinal + 1))
    report['sequential_gate'] = {'attempt': ordinal, 'alpha': alpha,
                                 'passed': report['paired_test']['p_value'] <= alpha}
    report['passed'] = report['passed'] and report['sequential_gate']['passed']
    report['quality_status'] = 'unknown' if not report['complete'] else 'passed' if report['passed'] else 'not_improved'
    return report


async def evaluate_outcomes(service, adapter, baseline=None):
    service._require_source_consent(adapter)
    run_id = adapter['training_run_id']
    run = service.store.get(service._kind(run_id), run_id)
    root = service.settings.data_dir / 'evaluations' / run_id / 'outcomes' / adapter['checkpoint_manifest_sha256']
    # Every distinct checkpoint gets a fresh, precommitted admission draw;
    # previous failures cannot be used as training targets for this draw.
    prompts = []
    dataset = run.get('dataset_path')
    if dataset and Path(dataset).is_file():
        import json
        for line in Path(dataset).read_text().splitlines():
            if line.strip():
                prompts.extend(turn['content'] for turn in json.loads(line).get('messages', []) if turn['role'] == 'user')
    binding = prepare_resources(root / 'resources', mode='sleep', training_prompts=prompts)
    attempts = dict(run.get('behavioral_attempts') or {})
    artifact = adapter['checkpoint_manifest_sha256']
    identity = {'resources': resource_identity(binding), 'baseline_adapter_id': baseline.get('id') if baseline else None,
                'baseline_artifact': baseline.get('checkpoint_manifest_sha256') if baseline else None}
    if artifact in attempts:
        previous = attempts[artifact]
        if {key: value for key, value in previous.items() if key != 'ordinal'} != identity:
            raise ValueError('Pinned Sleep outcome attempt baseline/resources changed')
    else:
        attempts[artifact] = {**identity, 'ordinal': len(attempts) + 1}
        service.store.update(service._kind(run_id), run_id, behavioral_attempts=attempts)
    service.store.update('adapters', adapter['id'], behavioral_resources=binding)
    service.store.update(service._kind(run_id), run_id, behavioral_resources=binding)
    async def progress(done, total, row):
        service._progress(run_id, stage='behavioral_evaluation', eval_completed=done, eval_total=total,
                          case_id=row['id'])
    try:
        before = await measure_selection(service, binding, root / 'baseline', model_id=adapter['base_model_id'],
                                         adapter=baseline, progress=progress)
        after = await measure_selection(service, binding, root / 'candidate', model_id=adapter['base_model_id'],
                                        adapter=adapter, progress=progress)
    finally:
        await service.manager.activate_adapter(baseline)
    service._require_source_consent(adapter)
    report = sequential_report(compare(load_resources(binding), before, after), attempts[artifact]['ordinal'])
    report['resources'] = resource_identity(binding)
    report['evidence_files'] = evidence_files(root / 'baseline', root / 'candidate')
    service.store.update(service._kind(run_id), run_id, behavioral_report=report)
    service.store.update('adapters', adapter['id'], behavioral_report=report)
    return report


def validate_admission(service, adapter, evaluation):
    """Activation rechecks the sealed task resources and actual episode receipts."""
    if not adapter.get('behavioral_resources'):
        return
    import json
    binding = adapter['behavioral_resources']
    verify_evidence((evaluation.get('behavioral') or {}).get('evidence_files'))
    root = Path(binding['path']).parent.parent
    report = compare(load_resources(binding), json.loads((root / 'baseline/measurement.json').read_text()),
                     json.loads((root / 'candidate/measurement.json').read_text()))
    run = service.store.get(service._kind(adapter['training_run_id']), adapter['training_run_id'])
    attempt = run['behavioral_attempts'][adapter['checkpoint_manifest_sha256']]
    if attempt['resources'] != resource_identity(binding):
        raise ValueError('Sleep sequential admission resources changed')
    report = sequential_report(report, attempt['ordinal'])
    report['resources'] = resource_identity(binding)
    report['evidence_files'] = evidence_files(root / 'baseline', root / 'candidate')
    if report != evaluation.get('behavioral') or not report['passed']:
        raise ValueError('Behavioral admission evidence changed or did not pass')


def purge_revoked_resources(store, run_id):
    """Repeat after owned jobs exit, so a revoked source cannot leave late writes."""
    if not isinstance(run_id, str) or Path(run_id).name != run_id:
        return
    try:
        run = store.get('sleep_runs' if run_id.startswith('sleep') else 'training_runs', run_id)
    except KeyError:
        return
    if not (run.get('artifact_revoked') or run.get('source_revoked') or run.get('source_invalidated')):
        return
    root = (store.path.parent / 'evaluations').resolve()
    directory = root / run_id
    if directory.is_symlink():
        directory.unlink()
    elif directory.is_dir() and directory.resolve().parent == root:
        shutil.rmtree(directory)
