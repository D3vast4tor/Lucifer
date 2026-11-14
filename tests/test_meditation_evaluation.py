import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lucifer.head_pack import file_hash
from lucifer.meditation_evaluation import (
    CONTRACT, EvaluationMemoryFailure, config_is_live, evaluate_with_fallback, evaluation_failure,
)


@pytest.fixture
def failed_evaluation(tmp_path):
    worker = Path(__file__).parents[1] / 'lucifer/meditation_worker.py'
    tree = ast.parse(worker.read_text())
    line = next(node.lineno for function in tree.body if isinstance(function, ast.FunctionDef)
                and function.name == '_resource_check' for node in ast.walk(function)
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                and any(key.arg == 'resource' and isinstance(key.value, ast.Constant)
                        and key.value.value == 'vram' for key in node.exc.keywords))
    root = tmp_path / 'owned'
    root.mkdir()
    paths = {}
    for name in ('head', 'reference', 'features', 'labels'):
        path = root / name
        path.write_bytes(name.encode())
        paths[name] = str(path)
    output = root / 'gpu-evaluate'
    output.mkdir()
    config = {'mode': 'evaluate', 'head_path': paths['head'], 'head_sha256': file_hash(paths['head']),
              'reference_head_path': paths['reference'], 'reference_head_sha256': file_hash(paths['reference']),
              'output_dir': str(output), 'datasets': {'development': {'features_path': paths['features'],
                  'features_sha256': file_hash(paths['features']), 'labels_path': paths['labels'],
                  'labels_sha256': file_hash(paths['labels'])}}}
    (output / 'worker_config.json').write_text(json.dumps(config))
    error = {'mode': 'evaluate', 'error_type': 'ResourceLimitError', 'frames': [
        {'file': str(worker.resolve()), 'function': '_resource_check', 'line': line}]}
    (output / 'error.json').write_text(json.dumps(error))
    run = {'native_runtime_binding': {'config': {'optimizer_worker_sha256': file_hash(worker)}}}
    return root, config, error, run, worker


def test_readonly_recovery_requires_exact_vram_frame_and_pinned_unchanged_inputs(failed_evaluation, monkeypatch):
    root, config, error, run, worker = failed_evaluation
    options = {'run': run, 'run_dir': root, 'worker_path': worker}
    receipt = evaluation_failure(config, error, **options)
    assert receipt['contract'] == CONTRACT and receipt['mutation_replayed'] is False
    assert evaluation_failure({**config, 'mode': 'train'}, error, **options) is None
    assert evaluation_failure(config, {**error, 'error_type': 'ValueError'}, **options) is None
    assert evaluation_failure(config, {**error, 'frames': []}, **options) is None
    monkeypatch.setattr('lucifer.meditation_evaluation.config_is_live', lambda path: True)
    assert evaluation_failure(config, error, **options) is None
    monkeypatch.setattr('lucifer.meditation_evaluation.config_is_live', lambda path: False)
    Path(config['head_path']).write_bytes(b'changed')
    assert evaluation_failure(config, error, **options) is None


def test_readonly_recovery_never_overwrites_a_committed_result_or_accepts_changed_code(failed_evaluation):
    root, config, error, run, worker = failed_evaluation
    options = {'run': run, 'run_dir': root, 'worker_path': worker}
    fake = {**run, 'native_runtime_binding': {'config': {'optimizer_worker_sha256': '0' * 64}}}
    assert evaluation_failure(config, error, **{**options, 'run': fake}) is None
    (Path(config['output_dir']) / 'result.json').write_text('{}')
    assert evaluation_failure(config, error, **options) is None


def test_recovery_observes_a_real_config_process_instead_of_a_marker_file(tmp_path):
    config = tmp_path / 'worker_config.json'
    config.write_text('{}')
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)', '--config', str(config)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert process.poll() is None and config_is_live(config)
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert not config_is_live(config)


@pytest.mark.asyncio
@pytest.mark.parametrize('delta', [0.000001, 0.01])
async def test_cpu_fallback_preserves_candidate_and_checks_baseline_before_selection(tmp_path, monkeypatch, delta):
    from test_meditation_service import harness, seed_run
    service, store, _ = harness(tmp_path)
    run = seed_run(service, store)
    store.update('meditation_runs', run['id'], evaluation_baseline={
        'head_path': 'baseline', 'head_sha256': 'baseline-sha', 'shape': [2, 2], 'dataset': {},
        'measurement': {'loss': 2.0, 'tokens': 8, 'group_losses': {'g': 2.0}}})
    calls = []
    async def worker(ident, config):
        calls.append(config)
        assert config['mode'] == 'evaluate'  # Never replay training.
        if config.get('device') != 'cpu':
            raise EvaluationMemoryFailure({'contract': CONTRACT})
        if 'evaluation-cpu-parity' in config['output_dir']:
            return {'head_sha256': 'baseline-sha', 'datasets': {'development': {
                'loss': 2.0 + delta, 'tokens': 8, 'group_losses': {'g': 2.0 + delta}}}}
        assert config['head_path'] == 'unchanged-candidate'
        return {'head_sha256': 'candidate-sha'}
    monkeypatch.setattr(service, '_worker', worker)
    config = {'mode': 'evaluate', 'head_path': 'unchanged-candidate',
              'output_dir': str(service._run_dir(run['id']) / 'development')}
    try:
        if delta > 1e-5:
            with pytest.raises(ValueError, match='pinned parity'):
                await evaluate_with_fallback(service, run['id'], config)
            assert len(calls) == 2
        else:
            result = await evaluate_with_fallback(service, run['id'], config)
            assert result['head_sha256'] == 'candidate-sha' and len(calls) == 3
            await evaluate_with_fallback(service, run['id'], config)
            assert len(calls) == 4  # No second GPU trial or parity measurement.
        assert store.get('meditation_runs', run['id'])['evaluation_device'] == 'cpu'
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_controller_resume_keeps_feature_and_probe_identity_after_evaluation_policy_change(tmp_path, monkeypatch):
    from lucifer.adaptive import atomic_json
    from lucifer.meditation_evaluation import completed_preparation, abandon_superseded_preparation
    from test_meditation_service import harness, seed_run, bundle, mocked_numerical_run
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    native = service._native_config(manager.specs['primary'])
    native['optimizer_support_sha256'] = {'meditation_calibration.py': 'old-policy'}
    monkeypatch.setattr(service, '_native_config', lambda spec: dict(native))
    original_worker = service._worker
    captured, calibrations, baselines = [], [], []

    async def worker(ident, config):
        if config['mode'] == 'prepare':
            captured.append(config)
            existing = completed_preparation(service, ident, config)
            if existing:
                assert config == existing['binding']['config']
                return json.loads((Path(config['output_dir']) / 'result.json').read_text())
            result = await original_worker(ident, config)
            atomic_json(Path(config['output_dir']) / 'result.json', result)
            store.put('meditation_operations', {'id': 'original-capture', 'phase': 'worker', 'status': 'completed',
                'binding': {'run_id': ident, 'config': config},
                'result_sha256': file_hash(Path(config['output_dir']) / 'result.json')})
            return result
        if config['mode'] == 'evaluate':
            baselines.append(config)
        assert config['mode'] != 'train'
        return await original_worker(ident, config)

    async def calibrate(service, ident, policy, **kwargs):
        calibrations.append(kwargs['feature_contract'])
        if len(calibrations) == 1:
            store.update('meditation_runs', ident,
                learning_rate_calibration={'pending_trial': {'completed_updates': 5}})
        else:
            assert store.get('meditation_runs', ident)['learning_rate_calibration']['pending_trial']['completed_updates'] == 5
        raise RuntimeError('test stops before optimizer effects')

    async def prepare(*args, **kwargs):
        return bundle()
    monkeypatch.setattr(service, '_worker', worker)
    monkeypatch.setattr('lucifer.meditation_calibration.calibrate_initial_rate', calibrate)
    try:
        await service._run(run['id'], prepare=prepare)
        native['optimizer_support_sha256'] = {'meditation_calibration.py': 'new-policy'}
        # Reconcile only the redundant, exited read-only capture. Its original
        # sealed predecessor remains intact; a changed dataset cannot qualify.
        redundant = {**captured[0], **native, 'output_dir': str(service._run_dir(run['id']) / 'redundant')}
        Path(redundant['output_dir']).mkdir()
        atomic_json(Path(redundant['output_dir']) / 'worker_config.json', redundant)
        effect = store.put('meditation_operations', {'id': 'redundant', 'phase': 'worker', 'status': 'prepared',
            'binding': {'run_id': run['id'], 'config': redundant}, 'started_at': 'observed-start'})
        assert completed_preparation(service, run['id'], {**redundant, 'dataset_sha256': 'changed'}) is None
        assert abandon_superseded_preparation(service, run['id'], effect)
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert len(calibrations) == 2 and calibrations[0] == calibrations[1]
        assert captured[0] == captured[1] and baselines[0] == baselines[1]
        assert result['native_runtime_binding']['config']['optimizer_support_sha256'] == native['optimizer_support_sha256']
        assert result['feature_capture_reuse']['optimizer_work_replayed'] is False
        # All actual capture arithmetic dependencies still invalidate reuse.
        for key in ('optimizer_worker_sha256', 'capture_worker_sha256', 'tokenizer_files', 'execution_plan_sha256'):
            assert completed_preparation(service, run['id'], {**captured[0], key: 'changed'}) is None
        Path(captured[0]['output_dir'], 'result.json').write_text('{}')
        with pytest.raises(ValueError, match='preparation changed'):
            completed_preparation(service, run['id'], captured[0])
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_explicit_resume_reconciles_readonly_failure_and_keeps_completed_probe(failed_evaluation, tmp_path, monkeypatch):
    from test_meditation_service import harness, seed_run
    root, config, error, evidence, worker = failed_evaluation
    service, store, _ = harness(tmp_path / 'service')
    # Use the fixture's owned root as this service's run directory; no process
    # or numerical worker is launched by the mocked continuation.
    monkeypatch.setattr(service, '_run_dir', lambda ident: root)
    run = seed_run(service, store, status='failed')
    store.update('meditation_runs', run['id'], **evidence, learning_rate_calibration={
        'pending_trial': {'completed_updates': 5, 'head_path': config['head_path'], 'head_sha256': config['head_sha256']}})
    store.put('meditation_operations', {'id': 'old-evaluation', 'phase': 'worker', 'status': 'prepared',
        'started_at': 'started', 'binding': {'run_id': run['id'], 'config': config}})
    calls = []
    async def continuation(ident):
        calls.append(ident)
    monkeypatch.setattr(service, '_run', continuation)
    try:
        before = file_hash(config['head_path'])
        await service.resume(run['id'], 'cpu-resume')
        await service.tasks[run['id']]
        result = store.get('meditation_runs', run['id'])
        assert result['evaluation_device'] == 'cpu'
        assert result['learning_rate_calibration']['pending_trial']['completed_updates'] == 5
        assert store.get('meditation_operations', 'old-evaluation')['status'] == 'abandoned_read_only'
        assert file_hash(config['head_path']) == before
        await service.resume(run['id'], 'cpu-resume')
        assert calls == [run['id']]
    finally:
        await service.close()
        store.close()
