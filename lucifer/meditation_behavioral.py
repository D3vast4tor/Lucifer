"""Native-head outcome evaluation under the controller's exclusive model lease."""
import asyncio
import json
from pathlib import Path

from .adaptive import atomic_json, file_sha256
from .behavioral import compare
from .behavioral_job import load_resources, measure_selection, prepare_resources, resource_identity, evidence_files, verify_evidence
from .store import now


def prepare(service, ident, bundle, *, attempt=None):
    from .behavioral_resources import attach_numeric_validation
    root = service._run_dir(ident) / 'behavioral'
    if attempt is not None:
        root = root / 'attempts' / str(attempt)
    prompts = [turn['content'] for row in bundle['examples'] for turn in row.get('messages', [])
               if turn['role'] == 'user']
    binding = prepare_resources(root / 'resources', mode='meditation', training_prompts=prompts)
    service.store.update('meditation_runs', ident, behavioral_resources=binding)
    return bundle if attempt is not None else attach_numeric_validation(bundle, binding['suite_sha256'])


async def admission(service, ident, head_path, exported, run_dir, *, numerical_retention=True):
    run = service.store.get('meditation_runs', ident)
    if not numerical_retention:
        report = {'passed': False, 'status': 'rejected_numerical_retention',
                  'numerical_retention_passed': False,
                  'behavioral_measured': False, 'reason': 'Numerical retention failed before serving the candidate'}
        service.store.update('meditation_runs', ident, admission_report=report)
        return report
    binding = run['behavioral_resources']
    root = Path(binding['path']).parent.parent
    parent = service.manager.specs[run['base_model_id']]
    native = service._native_config(parent)
    head_sha = file_sha256(head_path)
    # Seal the unadmitted head first. It can only load through the enclosing
    # job's private evaluation-profile exception, never general activation.
    provisional = {'passed': False, 'head_sha256': head_sha,
        'source_bundle_sha256': run['source_bundle_sha256'],
        'distribution_retention': run.get('distribution_retention')}
    candidate = await service._save_candidate(ident, head_path, exported, parent, native,
        Path(run['source_cache_path']), provisional, run['weight_update_receipt'])
    service.store.update('meditation_runs', ident, candidate_base_id=candidate['id'])
    pending = run.get('pending_behavioral_admission')
    attempt = pending['attempt'] if pending else run.get('behavioral_admission_attempts', 0) + 1
    intent = {'attempt': attempt, 'head_sha256': head_sha, 'resources': resource_identity(binding)}
    if pending and pending != intent:
        raise ValueError('Interrupted behavioral admission differs from its pinned checkpoint/resources')
    service.store.update('meditation_runs', ident, pending_behavioral_admission=intent,
                         behavioral_admission_attempts=attempt)
    parent_eval = parent
    if not parent.base_revision:
        # Use the identical controlled execution plan for original and updated
        # heads. Do not compare an aggressively offloaded parent to a candidate
        # running with different expert/thread/window settings.
        runtime = json.loads(Path(native['runtime_profile']).read_text())
        args = service._apply_execution_plan(runtime['args'], candidate['execution_plan'])
        if '--reserve-mib' in args:
            args[args.index('--reserve-mib') + 1] = str(max(2048, int(args[args.index('--reserve-mib') + 1])))
        if '--profile-readonly' not in args:
            args.append('--profile-readonly')
        runtime.update(args=args, args_written=list(args))
        profile = root / 'parent-runtime.json'
        atomic_json(profile, runtime)
        parent_eval = parent.model_copy(update={'id': ident + '-parent-evaluation',
            'env': {**parent.env, **candidate['execution_plan']['env'], 'LUCIFER_STRATA_PROFILE': str(profile)}})
        registered = service.manager.specs.get(parent_eval.id)
        if registered is not None and registered != parent_eval:
            raise ValueError('Owned behavioral parent profile changed')
        service.manager.register(parent_eval)
    async def progress(done, total, row):
        service._progress(ident, 'behavioral_evaluation', eval_completed=done, eval_total=total,
                          case_id=row['id'])
    try:
        service._progress(ident, 'behavioral_baseline_load')
        await service.manager.load(parent_eval.id)
        before = await measure_selection(service.learning, binding, root / 'baseline', model_id=parent_eval.id,
                                         artifact_sha256=exported['parent_head_sha256'], progress=progress)
        service._evaluation_profile = candidate['model_id']
        service._progress(ident, 'behavioral_candidate_load')
        await service.manager.load(candidate['model_id'])
        after = await measure_selection(service.learning, binding, root / 'candidate' / head_sha,
            model_id=candidate['model_id'], artifact_sha256=head_sha, progress=progress)
        report = compare(load_resources(binding), before, after)
        report['resources'] = resource_identity(binding)
        report['measurement_paths'] = {'baseline': str(root / 'baseline/measurement.json'),
            'candidate': str(root / 'candidate' / head_sha / 'measurement.json')}
        report['evidence_files'] = evidence_files(root / 'baseline', root / 'candidate' / head_sha)
        alpha = .05 / (attempt * (attempt + 1))
        sequential = {'attempt': attempt, 'alpha': alpha,
                      'passed': report['paired_test']['p_value'] <= alpha}
        passed = report['passed'] and numerical_retention and sequential['passed']
        final = {**provisional, 'passed': passed, 'behavioral': report,
                 'numerical_retention_passed': numerical_retention,
                 'sequential_gate': sequential,
                 'status': 'passed' if passed else 'rejected'}
        service._check_sources(json.loads(Path(run['source_cache_path']).read_text()))
        service.store.update('meditation_runs', ident, behavioral_report=report, admission_report=final,
                             pending_behavioral_admission=None)
        service.store.update('base_candidates', candidate['id'], admission_report=final,
                             status='candidate' if passed else 'rejected')
        admission_id = ident + '_behavioral_' + head_sha[:24]
        service.store.put('meditation_admissions', {'id': admission_id, 'run_id': ident,
            'source_conversation_ids': run['source_conversation_ids'], 'status': final['status'],
            'report': final, 'created_at': now()})
        if passed:
            pending = service._admission_snapshot(ident, head_path, exported, final, admission_id)
            service.store.update('meditation_runs', ident, pending_admission=pending)
        return final
    finally:
        service._evaluation_profile = None
        # Release owned evaluation profiles only after their server exits.
        await asyncio.shield(service.manager.unload())
        if parent_eval.id != parent.id:
            service.manager.specs.pop(parent_eval.id, None)
            service.store.delete('models', parent_eval.id)
            for name in ('_registry_sources', '_gpu_assignments', '_context_limits', '_native_context_cache', '_provider_adapters'):
                getattr(service.manager, name, {}).pop(parent_eval.id, None)
            registry = getattr(service.manager, 'capabilities', None)
            if registry is not None:
                registry.invalidate(parent_eval.id)


def validate_admission(service, candidate):
    from .meditation import digest
    bound = candidate.get('candidate_binding', {}).get('behavioral_resources')
    if not bound:
        return
    run = service.store.get('meditation_runs', candidate['training_run_id'])
    binding = run['behavioral_resources']
    if bound != resource_identity(binding):
        raise ValueError('Native behavioral resource binding changed')
    report = candidate['admission_report']['behavioral']
    verify_evidence(report.get('evidence_files'))
    paths = report['measurement_paths']
    before, after = [json.loads(Path(paths[key]).read_text()) for key in ('baseline', 'candidate')]
    if after.get('selection', {}).get('selected_artifact_sha256') != candidate['head_sha256']:
        raise ValueError('Behavioral outcomes belong to another native head')
    checked = compare(load_resources(binding), before, after)
    checked.update(resources=resource_identity(binding), measurement_paths=paths)
    checked['evidence_files'] = evidence_files(Path(paths['baseline']).parent, Path(paths['candidate']).parent)
    if digest(checked) != digest(report) or not checked['passed']:
        raise ValueError('Native behavioral evaluation changed or failed')
    sequential = candidate['admission_report'].get('sequential_gate')
    if sequential:
        alpha = .05 / (sequential['attempt'] * (sequential['attempt'] + 1))
        if sequential != {'attempt': sequential['attempt'], 'alpha': alpha,
                          'passed': checked['paired_test']['p_value'] <= alpha} or not sequential['passed']:
            raise ValueError('Native sequential admission confidence failed or changed')
