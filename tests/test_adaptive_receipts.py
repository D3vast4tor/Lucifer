"""Training receipt recovery uses stable hardware and immutable executed inputs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lucifer.adaptive import AdaptiveState, commit_checkpoint
from lucifer.adaptive_learning import EvidenceRequired, digest, worker_segment
from lucifer.store import Store


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / 'runtime'
    root.mkdir()
    (root / 'datasets').mkdir()
    store = Store(root / 'test.db')
    run_id = 'sleep_receipt'
    request = {'method': 'qlora', 'rank': 8, 'batch_size': 1, 'gradient_accumulation': 4,
               'learning_rate': .0001, 'max_seq_length': 4096, 'dynamic': True}
    spec = {'id': 'student', 'path': 'org/student', 'training_model': 'org/student'}
    revision = {'sha256': 'a' * 64, 'dataset_path': str(root / 'datasets' / 'reviewed.jsonl')}
    state = AdaptiveState.initial(seed=19)
    placement = {'device_count': 1, 'backend': 'cuda', 'devices': [
        {'id': 'GPU-stable', 'backend': 'cuda', 'index': '0', 'used_mb': 1800.,
         'free_mb': 10000., 'utilization_pct': 0., 'temperature_c': 40., 'power_w': 10.}],
        'env': {'CUDA_VISIBLE_DEVICES': '0'}}
    service = SimpleNamespace(store=store, settings=SimpleNamespace(data_dir=root),
                              manager=SimpleNamespace(gpu_inventory={'gpus': []}), processes={},
                              _progress=lambda *args, **kwargs: None)
    store.put('sleep_runs', {'id': run_id})
    monkeypatch.setattr('lucifer.gpu.training_placement', lambda *args, **kwargs: copy.deepcopy(placement))
    async def forbidden(*args, **kwargs):
        raise AssertionError('Receipt recovery must not launch training')
    monkeypatch.setattr('lucifer.adaptive_learning.asyncio.create_subprocess_exec', forbidden)
    yield SimpleNamespace(root=root, store=store, service=service, run_id=run_id, request=request,
                          spec=spec, revision=revision, state=state, placement=placement)
    store.close()


def legacy_receipt(env, *, status='started', config=True, temperature=20.):
    old_placement = copy.deepcopy(env.placement)
    old_placement.pop('env')
    old_placement['devices'][0].update(used_mb=11000., free_mb=800., utilization_pct=100.,
                                      temperature_c=temperature, power_w=100.)
    binding = {'state': digest(env.state.to_dict()), 'data_revision': env.revision['sha256'],
               'resume_sha256': None, 'placement': old_placement}
    fingerprint = digest([env.run_id, 'training_segment', binding])
    ident = 'adaptive_op_' + fingerprint[:32]
    operation = env.store.put('adaptive_operations', {'id': ident, 'run_id': env.run_id,
        'phase': 'training_segment', 'binding': binding, 'fingerprint': fingerprint, 'status': status})
    output = env.root / 'adaptive' / env.run_id / ident
    checkpoint = output / 'checkpoints' / 'checkpoint-6'
    checkpoint.mkdir(parents=True)
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt',
                 'training_metrics.json', 'rng_state.pth'):
        (checkpoint / name).write_bytes(b'owned-checkpoint')
    (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 6, 'epoch': 1.}))
    end = AdaptiveState.from_dict(env.state.to_dict())
    end.consume_updates(6)
    end.dataset_passes_completed = 1
    identity = {'base_model': 'org/student', **{key: env.request[key] for key in
        ('method', 'rank', 'batch_size', 'gradient_accumulation', 'learning_rate')}}
    manifest = commit_checkpoint(checkpoint, {'segment_id': ident, 'data_revision': env.revision['sha256'],
        'parent_manifest_sha256': None, 'state': end.to_dict(), 'world_size': 1, 'training_identity': identity})
    result = {'segment_id': ident, 'checkpoint_path': str(checkpoint), 'checkpoint_manifest_sha256': manifest,
              'data_revision': env.revision['sha256'], 'state': end.to_dict(), 'world_size': 1,
              'boundary_reasons': ['dataset_pass']}
    result_path = output / 'segment_result.json'
    result_path.write_text(json.dumps(result))
    if status == 'completed':
        operation = env.store.update('adaptive_operations', ident,
                                    result={'result_path': str(result_path), 'sha256': digest(result)})
    if config:
        settings = {**env.request, 'base_model': 'org/student', 'adaptive': {'state': env.state.to_dict(),
                    'data_revision': env.revision['sha256'], 'resume_manifest_sha256': None, 'segment_id': ident}}
        (env.root / 'datasets' / (ident + '.training.json')).write_text(json.dumps(settings))
    return operation, result


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['started', 'completed'])
async def test_legacy_completed_files_reused_despite_changed_live_gpu_counters(env, status):
    operation, expected = legacy_receipt(env, status=status)
    result = await worker_segment(env.service, env.run_id, env.request, env.spec, env.revision, env.state, None)
    assert result == expected
    final = env.store.get('adaptive_operations', operation['id'])
    assert final['status'] == 'completed'
    assert final['binding'] == operation['binding'] and final['fingerprint'] == operation['fingerprint']
    assert len(env.store.list('adaptive_operations')) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('key,value', [('learning_rate', .0002), ('max_seq_length', 8192), ('dynamic', False)])
async def test_legacy_execution_cannot_be_reused_or_retrained_with_different_config(env, key, value):
    operation, _ = legacy_receipt(env)
    request = {**env.request, key: value}
    with pytest.raises(EvidenceRequired, match='configuration does not match'):
        await worker_segment(env.service, env.run_id, request, env.spec, env.revision, env.state, None)
    assert env.store.get('adaptive_operations', operation['id'])['status'] == 'started'
    assert len(env.store.list('adaptive_operations')) == 1


@pytest.mark.asyncio
async def test_legacy_config_missing_fails_closed_without_new_training(env):
    legacy_receipt(env, config=False)
    with pytest.raises(EvidenceRequired, match='original training configuration is unavailable'):
        await worker_segment(env.service, env.run_id, env.request, env.spec, env.revision, env.state, None)


@pytest.mark.asyncio
async def test_multiple_potential_legacy_executions_pause_instead_of_guessing(env):
    legacy_receipt(env, temperature=20.)
    legacy_receipt(env, temperature=30.)
    with pytest.raises(EvidenceRequired, match='Multiple training operations'):
        await worker_segment(env.service, env.run_id, env.request, env.spec, env.revision, env.state, None)


@pytest.mark.asyncio
async def test_legacy_original_fingerprint_must_verify(env):
    operation, _ = legacy_receipt(env)
    env.store.update('adaptive_operations', operation['id'], fingerprint='changed')
    with pytest.raises(EvidenceRequired, match='identity changed'):
        await worker_segment(env.service, env.run_id, env.request, env.spec, env.revision, env.state, None)


@pytest.mark.asyncio
async def test_new_binding_excludes_live_counters_and_includes_execution_hyperparameters(env, monkeypatch):
    class Output:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration
    class FailedMockWorker:
        stdout = Output()
        async def wait(self):
            return 1
    async def mocked(*args, **kwargs):
        return FailedMockWorker()
    monkeypatch.setattr('lucifer.adaptive_learning.asyncio.create_subprocess_exec', mocked)
    with pytest.raises(RuntimeError, match='worker failed'):
        await worker_segment(env.service, env.run_id, env.request, env.spec, env.revision, env.state, None)
    operation, = env.store.list('adaptive_operations')
    assert operation['binding']['placement'] == {'device_count': 1, 'backend': 'cuda',
                                                'devices': [{'id': 'GPU-stable', 'backend': 'cuda'}]}
    assert operation['binding']['training_identity'] == {'base_model': 'org/student', **env.request}
    assert operation['fingerprint'] == digest([env.run_id, 'training_segment', operation['binding']])


@pytest.mark.asyncio
async def test_subprocess_heartbeats_use_successful_counts_until_sealed_result_adoption(env, monkeypatch):
    from lucifer.adaptive_learning import save_state
    env.state.consume_updates(18)
    committed = env.state.to_dict()
    live = {'optimizer_step': 18, 'cumulative_optimizer_steps': 18, 'committed_optimizer_steps': 18,
            'round_remaining': 32}
    updates = []
    sealed = False
    def progress(run_id, **values):
        assert run_id == env.run_id
        if not sealed:
            assert env.state.to_dict() == committed
            assert env.store.get('sleep_runs', run_id).get('adaptive') is None
        live.update(values)
        updates.append(copy.deepcopy(values))
        env.store.update('sleep_runs', run_id, progress=copy.deepcopy(live))
    env.service._progress = progress

    class Output:
        def __init__(self, segment_id):
            def heartbeat(count, *, skipped=False, epoch=3.5):
                return {'event': 'adaptive_progress', 'segment_id': segment_id,
                        'global_successful_updates': count, 'segment_successful_updates': count - 18,
                        'round_remaining': 50 - count, 'randint_min': 1, 'randint_max': 7,
                        'next_refresh_at': 50, 'native_epoch': epoch, 'optimizer_skipped': skipped}
            native = {'event': 'training_progress', 'step': 999, 'total_steps': 1000, 'epoch': 3.25, 'loss': .8}
            final_native = {**native, 'step': 1000, 'epoch': 4.0, 'loss': .7}
            self.lines = iter([json.dumps(native).encode(),
                               ('\r TQDM ' + json.dumps(heartbeat(19))).encode(),
                               json.dumps(heartbeat(19, skipped=True)).encode(),
                               json.dumps(heartbeat(20, epoch=4.0)).encode(),
                               json.dumps(final_native).encode()])
        def __aiter__(self):
            return self
        async def __anext__(self):
            try:
                return next(self.lines)
            except StopIteration:
                raise StopAsyncIteration

    class MockWorker:
        def __init__(self, config):
            self.config = config
            self.stdout = Output(config['adaptive']['segment_id'])
        async def wait(self):
            nonlocal sealed
            # Every stdout heartbeat was observed before checkpoint publication.
            assert live['cumulative_optimizer_steps'] == live['optimizer_step'] == 20
            assert live['round_remaining'] == 30
            assert live['committed_optimizer_steps'] == 18
            assert live['native_optimizer_step'] == 1000
            assert env.state.to_dict() == committed
            assert not (Path(self.config['output_dir']) / 'segment_result.json').exists()
            output = Path(self.config['output_dir'])
            checkpoint = output / 'checkpoints' / 'checkpoint-1000'
            checkpoint.mkdir(parents=True)
            for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt',
                         'training_metrics.json', 'rng_state.pth'):
                (checkpoint / name).write_bytes(b'owned-native-state')
            (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 1000, 'epoch': 4.0}))
            final = AdaptiveState.from_dict(committed)
            final.consume_updates(2)
            final.dataset_passes_completed += 1
            identity = {'base_model': 'org/student', **{key: env.request[key] for key in
                ('method', 'rank', 'batch_size', 'gradient_accumulation', 'learning_rate')}}
            manifest = commit_checkpoint(checkpoint, {'segment_id': self.config['adaptive']['segment_id'],
                'data_revision': env.revision['sha256'], 'parent_manifest_sha256': None,
                'state': final.to_dict(), 'world_size': 1, 'training_identity': identity})
            result = {'segment_id': self.config['adaptive']['segment_id'], 'checkpoint_path': str(checkpoint),
                      'checkpoint_manifest_sha256': manifest, 'data_revision': env.revision['sha256'],
                      'state': final.to_dict(), 'world_size': 1, 'boundary_reasons': ['dataset_pass']}
            (output / 'segment_result.json').write_text(json.dumps(result))
            sealed = True
            return 0

    async def launch(*args, **kwargs):
        return MockWorker(json.loads(Path(args[-1]).read_text()))
    monkeypatch.setattr('lucifer.adaptive_learning.asyncio.create_subprocess_exec', launch)
    result = await worker_segment(env.service, env.run_id, env.request, env.spec, env.revision, env.state, None)
    successful = [values for values in updates if 'cumulative_optimizer_steps' in values]
    assert [values['cumulative_optimizer_steps'] for values in successful] == [19, 19, 20]
    assert [values['round_remaining'] for values in successful] == [31, 31, 30]
    assert [values['optimizer_skipped'] for values in successful] == [False, True, False]
    assert all(values['committed_optimizer_steps'] == 18 for values in successful)
    native = [values for values in updates if 'native_optimizer_step' in values]
    assert [values['native_optimizer_step'] for values in native] == [999, 1000]
    assert all('optimizer_step' not in values and 'cumulative_optimizer_steps' not in values
               and 'committed_optimizer_steps' not in values for values in native)
    assert env.state.to_dict() == committed and live['committed_optimizer_steps'] == 18
    assert result['state']['global_successful_updates'] == 20
    # Only controller adoption of the verified result advances durable state.
    env.state = AdaptiveState.from_dict(result['state'])
    save_state(env.service, env.run_id, {}, env.state)
    assert live['committed_optimizer_steps'] == live['cumulative_optimizer_steps'] == live['optimizer_step'] == 20
    assert env.store.get('sleep_runs', env.run_id)['adaptive']['state'] == result['state']
