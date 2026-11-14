"""Orchestration contracts with tiny files and mocked numerical workers.

These fixtures establish lifecycle behavior, not native-model training success.
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lucifer.adaptive import atomic_json, file_sha256
from lucifer.config import Settings
from lucifer.inference import ModelSpec
from lucifer.meditation import MeditationService, digest
from lucifer.meditation_policy import MeditationPolicy
from lucifer.store import Store


class Manager:
    def __init__(self, parent):
        self.specs = {parent.id: parent}
        self.selected = {'id': parent.id, 'adapter_id': None}
        self.loads = []
        self.adapters = []
        self.failure = None
        self.answer = None
        self.answers_by_model = {}

    def current(self):
        return dict(self.selected) if self.selected else None

    async def load(self, ident):
        self.loads.append(ident)
        if self.failure == ident:
            raise RuntimeError('owned loader failed')
        validator = getattr(self, 'native_base_validator', None)
        if validator is not None:
            validator(self.specs[ident])
        self.selected = {'id': ident, 'adapter_id': None}
        return self.current()

    async def unload(self):
        self.selected = None

    async def activate_adapter(self, adapter):
        self.adapters.append(adapter.get('id') if adapter else None)
        if self.selected is not None:
            self.selected['adapter_id'] = adapter.get('id') if adapter else None
        return self.current()

    def register(self, spec):
        self.specs[spec.id] = spec
        return spec.model_dump()

    async def complete(self, messages, model_id=None, adapter=None, temperature=.7,
                       max_tokens=512, thinking=None):
        if model_id in self.answers_by_model:
            return self.answers_by_model[model_id]
        if self.answer is not None:
            return self.answer
        prompt = messages[-1]['content']
        if '31 + 42' in prompt:
            return '73'
        if 'key "ready"' in prompt:
            return json.dumps({'ready': True})
        if 'tool read_file' in prompt:
            return json.dumps({'tool': 'read_file', 'arguments': {'path': 'notes.txt'}})
        if 'exactly STOP once' in prompt:
            return 'STOP'
        return 'No'


def harness(tmp_path):
    settings = Settings(data_dir=tmp_path / 'state', models_file=tmp_path / 'models.yaml')
    store = Store(settings.data_dir / 'state.sqlite')
    parent = ModelSpec(id='primary', backend='strata', path=str(tmp_path / 'pack'), runtime_command=['/bin/true'])
    manager = Manager(parent)
    learning = SimpleNamespace(lock=asyncio.Lock(), busy=False)
    service = MeditationService(settings, store, manager, learning)
    # Source-contract validation is exercised with real consented records in
    # test_meditation_sources. These orchestration rows are intentionally fake;
    # mock just the source commit adapter, retaining real immutable tiny files.
    def commit_sources(ident, run_dir, value, references):
        revision_sha = digest(value)
        directory = run_dir / 'source-revisions' / revision_sha
        directory.mkdir(parents=True, exist_ok=True)
        source_path, dataset_path = directory / 'sources.json', directory / 'dataset.json'
        atomic_json(source_path, value)
        datasets = {'train': value['examples'], 'development': value['development_examples'],
                    'holdout': value['holdout_examples']}
        atomic_json(dataset_path, {'datasets': datasets})
        ref = {'source_cache_path': str(source_path), 'source_bundle_sha256': file_sha256(source_path),
               'source_manifest_sha256': revision_sha, 'dataset_path': str(dataset_path),
               'dataset_sha256': file_sha256(dataset_path)}
        previous = references[-1].get('dataset_sha256') if references else None
        refs = list(references)
        if not any(old.get('source_bundle_sha256') == ref['source_bundle_sha256'] for old in refs):
            refs.append(ref)
        return {**ref, 'source_revision_refs': refs,
                'data_refreshed': previous is not None and previous != ref['dataset_sha256']}
    service._commit_sources = commit_sources
    return service, store, manager


def bundle():
    def row(group):
        return {'messages': [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'Answer'}],
                'meditation_group_id': group, 'fingerprint': group}
    return {'examples': [row('train')], 'development_examples': [row('dev')],
            'holdout_examples': [row('heldout')], 'source_conversation_ids': ['chat'],
            'manifest': {'fixtures_only': True}, 'metrics': {'train': 1, 'development': 1, 'heldout': 1}}


def seed_run(service, store, *, ident='meditation_fixture', dry_run=False, status='queued'):
    return store.put('meditation_runs', {'id': ident, 'status': status, 'base_model_id': 'primary',
        'request': {'idempotency_key': 'run', 'mode': 'supervised', 'dry_run': dry_run}, 'policy': MeditationPolicy().snapshot(),
        'source_conversation_ids': ['chat'], 'progress': {'stage': 'queued'}})


def candidate_fixture(service, store, manager, tmp_path):
    source_path = tmp_path / 'sources.json'
    atomic_json(source_path, bundle())
    head = tmp_path / 'head.bin'
    head.write_bytes(b'fixture-head')
    spec = manager.specs['primary'].model_copy(update={'id': 'changed', 'path': str(tmp_path / 'changed-pack'),
                                                     'base_revision': file_sha256(head),
                                                     'parent_model_id': 'primary'})
    manager.register(spec)
    return store.put('base_candidates', {'id': 'base_fixture', 'model_id': spec.id,
        'parent_model_id': 'primary', 'status': 'admitted', 'head_sha256': file_sha256(head),
        'source_cache_path': str(source_path), 'source_bundle_sha256': file_sha256(source_path),
        'source_conversation_ids': ['chat'], 'admission_report': {'passed': True},
        'spec': spec.model_dump(),
        'rollback_receipt': {'verified': True}, 'artifact_files': [{'path': str(head), 'sha256': file_sha256(head)}]})


@pytest.mark.asyncio
async def test_start_reuses_intent_and_dry_run_never_launches_worker(tmp_path, monkeypatch):
    import lucifer.meditation_sources as sources
    service, store, manager = harness(tmp_path)
    prepared = []
    async def prepare(*args, **kwargs):
        prepared.append(kwargs['run_id'])
        return bundle()
    async def forbidden(*args, **kwargs):
        pytest.fail('A dry run must not launch numerical workers')
    monkeypatch.setattr(sources, 'prepare_meditation_sources', prepare)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    monkeypatch.setattr(service, '_worker', forbidden)
    try:
        request = {'idempotency_key': 'one-intent', 'mode': 'supervised', 'dry_run': True, 'conversation_ids': ['chat']}
        first = await service.start(request)
        repeated = await service.start(request)
        assert first['id'] == repeated['id']
        await service.tasks[first['id']]
        assert len(prepared) == 1
        assert store.get('meditation_runs', first['id'])['status'] == 'dry_run'
        assert manager.current()['id'] == 'primary'
        with pytest.raises(ValueError, match='different Meditation request'):
            await service.start({**request, 'conversation_ids': ['other']})
        manager.specs['compact'] = ModelSpec(id='compact', backend='demo')
        with pytest.raises(ValueError, match='native primary'):
            await service.start({'idempotency_key': 'other', 'base_model': 'compact', 'dry_run': True})
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_resume_refuses_uncertain_worker_and_deduplicates_completed_intent(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store, status='interrupted')
    uncertain = store.put('meditation_operations', {'id': 'uncertain', 'phase': 'worker', 'status': 'prepared',
        'binding': {'run_id': run['id'], 'config': {'mode': 'train',
            'output_dir': str(service._run_dir(run['id']) / 'unsealed-worker')}}, 'started_at': 'before-crash'})
    calls, finish = [], asyncio.Event()
    async def resumed(ident):
        calls.append(ident)
        await finish.wait()
    monkeypatch.setattr(service, '_run', resumed)
    try:
        with pytest.raises(ValueError, match='uncertain effects'):
            await service.resume(run['id'], 'resume-one')
        assert not calls
        store.update('meditation_operations', uncertain['id'], status='completed')
        result = await service.resume(run['id'], 'resume-one')
        await asyncio.sleep(0)
        assert await service.resume(run['id'], 'resume-one') == result
        assert calls == [run['id']]
        finish.set()
        await service.tasks[run['id']]
    finally:
        finish.set()
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_worker_reuses_only_hash_bound_committed_result(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    output = service._run_dir(run['id']) / 'segment-1'
    output.mkdir()
    config = {'mode': 'train', 'output_dir': str(output), 'steps': 25}
    result = output / 'result.json'
    atomic_json(result, {'successful_updates': 25})
    key = digest([run['id'], config])
    operation = service._receipt('worker', key, {'run_id': run['id'], 'config': config})
    store.update('meditation_operations', operation['id'], status='completed', result_sha256=file_sha256(result))
    async def forbidden(*args, **kwargs):
        pytest.fail('A committed worker must not be launched again')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbidden)
    try:
        assert await service._worker(run['id'], config) == {'successful_updates': 25}
        result.write_text('{}')
        with pytest.raises(ValueError, match='changed or disappeared'):
            await service._worker(run['id'], config)
    finally:
        await service.close()
        store.close()


def test_candidate_requires_admission_rollback_current_sources_and_intact_artifacts(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    candidate = candidate_fixture(service, store, manager, tmp_path)
    try:
        service._verify_base(candidate)
        with pytest.raises(ValueError, match='passed evaluation'):
            service._verify_base({**candidate, 'rollback_receipt': {'verified': False}})
        with pytest.raises(ValueError, match='passed evaluation'):
            service._verify_base({**candidate, 'admission_report': {'passed': False}})
        with pytest.raises(ValueError, match='revoked'):
            service._verify_base({**candidate, 'source_revoked': True})
        def invalid(value):
            raise ValueError('Consumed source consent changed')
        monkeypatch.setattr(service, '_check_sources', invalid)
        with pytest.raises(ValueError, match='source consent'):
            service._verify_base(candidate)
        monkeypatch.setattr(service, '_check_sources', lambda value: None)
        Path(candidate['artifact_files'][0]['path']).write_bytes(b'tampered')
        with pytest.raises(ValueError, match='weights, manifest or launcher changed'):
            service._verify_base(candidate)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_activate_and_rollback_preserve_parent_selection_and_reuse_receipts(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    candidate = candidate_fixture(service, store, manager, tmp_path)
    try:
        saved = store.get('base_candidates', candidate['id'])
        assert saved['status'] == 'admitted' and manager.current()['id'] == 'primary'
        active = await service.activate(candidate['id'], 'activate-one')
        assert active['status'] == 'active'
        assert manager.current()['id'] == 'changed'
        assert store.setting('previous_base_id') == 'primary'
        assert store.setting('active_base_revision') == candidate['id']
        loads = list(manager.loads)
        assert (await service.activate(candidate['id'], 'activate-one'))['status'] == 'active'
        assert manager.loads == loads
        result = await service.rollback('rollback-one')
        assert result['model_id'] == 'primary'
        assert manager.current()['id'] == 'primary'
        assert store.setting('active_base_revision') is None
        assert store.get('base_candidates', candidate['id'])['status'] == 'archived'
        loads = list(manager.loads)
        assert await service.rollback('rollback-one') == result
        assert manager.loads == loads
        assert store.get('base_candidates', candidate['id'])['artifact_files']
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_failed_activation_restores_parent_without_publishing_active_revision(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    candidate = candidate_fixture(service, store, manager, tmp_path)
    manager.failure = 'changed'
    try:
        with pytest.raises(RuntimeError, match='owned loader failed'):
            await service.activate(candidate['id'], 'activate-failed')
        assert manager.current()['id'] == 'primary'
        assert store.setting('active_base_revision') is None
        assert store.get('base_candidates', candidate['id'])['status'] == 'admitted'
    finally:
        await service.close()
        store.close()


def test_checkpoint_pruning_keeps_current_best_and_never_follows_symlink(tmp_path):
    service, store, manager = harness(tmp_path)
    try:
        root = service._run_dir('meditation_prune')
        for name in ('segment-1', 'segment-2', 'segment-3'):
            (root / name).mkdir()
            (root / name / 'checkpoint').write_bytes(b'fixture')
        external = tmp_path / 'external'
        external.mkdir()
        (external / 'untouched').write_bytes(b'keep')
        (root / 'segment-linked').symlink_to(external, target_is_directory=True)
        service._prune_segments(root, {str(root / 'segment-2/checkpoint'), str(root / 'segment-3/checkpoint')})
        assert not (root / 'segment-1').exists()
        assert (root / 'segment-2/checkpoint').exists() and (root / 'segment-3/checkpoint').exists()
        assert (external / 'untouched').read_bytes() == b'keep'
    finally:
        store.close()


def mocked_numerical_run(service, store, manager, tmp_path, monkeypatch):
    """Tiny deterministic numerical receipts to exercise orchestration only."""
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    atomic_json(tmp_path / 'runtime.json', {'exe': '/bin/true', 'args': ['--pack', manager.specs['primary'].path],
                                         'tokenizer': str(tmp_path)})
    monkeypatch.setattr(service, '_native_config', lambda spec: {'engine': '/bin/true', 'pack': spec.path,
        'tokenizer_path': str(tmp_path), 'runtime_profile': str(tmp_path / 'runtime.json')})
    monkeypatch.setattr('lucifer.meditation.shutil.disk_usage', lambda path: SimpleNamespace(free=100_000_000_000))
    parent_head = tmp_path / 'parent-head.bin'
    parent_head.write_bytes(b'original-fixture')
    monkeypatch.setattr('lucifer.head_pack.load_pack', lambda path: SimpleNamespace(manifest_sha256='fixture-manifest'))
    monkeypatch.setattr('lucifer.head_pack.tensor_hash', lambda pack, name: file_sha256(parent_head))
    groups = [f'group-{i}' for i in range(8)]
    calls = []
    async def worker(ident, config):
        calls.append(config['mode'])
        output = Path(config['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        if config['mode'] == 'export':
            return {'head_path': str(parent_head), 'shape': [2, 2], 'parent_head_sha256': file_sha256(parent_head),
                    'bf16_head_sha256': file_sha256(parent_head), 'parent_manifest_sha256': 'fixture-manifest'}
        if config['mode'] == 'prepare':
            datasets = {name: {'groups': groups, 'native_nll': 2.0, 'count': 8, 'rows': 8,
                              'native_group_losses': {group: 2.0 for group in groups},
                              'native_group_clipped_losses': {group: 2.0 for group in groups},
                              'original_native_nll': 1.98,
                              'original_native_group_losses': {group: 1.98 for group in groups},
                              'original_native_group_clipped_losses': {group: 1.98 for group in groups},
                              'features_path': str(output / 'features'), 'labels_path': str(output / 'labels')}
                        for name in ('train', 'development', 'holdout')}
            baseline = {'datasets': {name: {'loss': 2.0, 'groups': groups,
                                          'group_losses': {group: 2.0 for group in groups},
                                          'group_clipped_losses': {group: 2.0 for group in groups}}
                                     for name in datasets}}
            original = {name: {'loss': 1.98, 'group_losses': {group: 1.98 for group in groups},
                              'group_clipped_losses': {group: 1.98 for group in groups}}
                        for name in datasets}
            return {'datasets': datasets, 'parity': {'passed': True}, 'native_gates': {'passed': True},
                    'baseline_bf16': baseline, 'baseline_original': original}
        if config['mode'] == 'evaluate':
            loss = 1.0 if len(config['datasets']) == 1 else 2.0
            return {'head_sha256': file_sha256(config['head_path']), 'datasets': {name: {'loss': loss, 'groups': groups,
                                       'group_losses': {group: loss for group in groups},
                                       'group_clipped_losses': {group: loss for group in groups}}
                                 for name in config['datasets']}}
        if config['mode'] == 'native_admission':
            selected = config['selected_groups']
            return {'mode': 'native_admission', 'backend': 'native_engine_teacher_forced',
                    'head_sha256': config['head_sha256'],
                    'group_clipped_losses': {group: 1.0 for group in selected},
                    'converted_group_clipped_losses': {group: 2.0 for group in selected},
                    'original_group_clipped_losses': {group: 1.98 for group in selected}}
        if config['mode'] == 'train':
            checkpoint = output / 'checkpoint'
            checkpoint.mkdir()
            head = checkpoint / 'head.bf16'
            head.write_bytes(b'updated-fixture')
            previous_updates = store.get('meditation_runs', ident)['policy']['successful_updates']
            return {'head_path': str(head), 'checkpoint_path': str(checkpoint),
                    'successful_updates': previous_updates + config['steps'],
                    'segment_updates': config['steps'], 'changed_elements': 4, 'changed_weight_count': 4,
                    'after_head_sha256': file_sha256(head), 'delta_l2': 0.5, 'delta_max_abs': .1}
        if config['mode'] == 'verify':
            return {'mode': 'verify', 'passed': True, 'rollback_probe_passed': True,
                    'updated_head_sha256': config['head_sha256'], 'logits_changed_elements': 4,
                    'original_head_unchanged': True, 'engine_sha256': file_sha256('/bin/true')}
        raise AssertionError(config['mode'])
    async def save(ident, head_path, exported, parent, native, bundle_path, report, trained):
        try:
            existing = store.get('base_candidates', 'base_fixture')
        except KeyError:
            existing = None
        if existing is not None:
            assert existing['head_sha256'] == file_sha256(head_path)
            assert existing['source_bundle_sha256'] == file_sha256(bundle_path)
            return existing
        spec = parent.model_copy(update={'id': 'changed', 'path': str(Path(head_path).parent / 'candidate-pack'),
                                         'base_revision': file_sha256(head_path),
                                         'parent_model_id': parent.id})
        manager.register(spec)
        return store.put('base_candidates', {'id': 'base_fixture', 'model_id': 'changed', 'parent_model_id': parent.id,
            'status': 'candidate', 'training_run_id': ident, 'admission_report': report, 'head_sha256': file_sha256(head_path),
            'source_conversation_ids': ['chat'], 'source_cache_path': str(bundle_path),
            'spec': spec.model_dump(),
            'source_bundle_sha256': file_sha256(bundle_path), 'artifact_files': [{'path': head_path, 'sha256': file_sha256(head_path)}]})
    monkeypatch.setattr(service, '_worker', worker)
    monkeypatch.setattr(service, '_save_candidate', save)
    return calls


@pytest.mark.asyncio
async def test_first_passing_checkpoint_is_saved_only_after_candidate_load_and_parent_restore(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    calls = mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    async def prepare(*args, **kwargs):
        return bundle()
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['status'] == 'admitted', result.get('error')
        assert calls.count('train') == 4  # Three disposable probes, one retained segment.
        assert result['learning_rate_calibration']['total_successful_updates'] == 15
        assert result['policy']['successful_updates'] == 25
        saved = store.get('base_candidates', 'base_fixture')
        assert saved['status'] == 'admitted' and saved['rollback_receipt']['verified'] is True
        assert result['saved_base_id'] == saved['id']
        assert manager.current()['id'] == 'primary'
        assert 'changed' in manager.loads and manager.loads[-1] == 'primary'
        assert all(model == 'primary' for model in manager.loads[manager.loads.index('changed') + 1:])
        assert store.setting('active_base_revision') is None
        assert result['progress']['restored'] is True
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['passing', 'missing', 'excessive'])
async def test_new_training_precision_requires_bound_distribution_retention_before_admission(tmp_path, monkeypatch, drift):
    from lucifer.meditation_retention import CONTRACT
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    store.update('meditation_runs', run['id'], training_precision='native_f32', retention_contract=CONTRACT)
    calls = mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    original = service._worker
    async def worker(ident, config):
        if config['mode'] == 'train':
            assert config['compute_precision'] == 'native_f32'
        result = await original(ident, config)
        if config['mode'] == 'evaluate' and config.get('reference_head_path'):
            result['reference_head_sha256'] = config['reference_head_sha256']
            if drift != 'missing':
                result['datasets']['development'].update(tokens=8, forward_kl={
                    'tokens': 8, 'mean': 0.001, 'maximum': 0.002 if drift == 'passing' else 0.1,
                    'direction': 'untrained_bf16_reference_to_candidate'})
        return result
    monkeypatch.setattr(service, '_worker', worker)
    async def prepare(*args, **kwargs):
        return bundle()
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['progress']['restored'] is True
        if drift == 'passing':
            assert result['status'] == 'admitted', result.get('error')
            receipt = result['admission_report']['distribution_retention']
            assert receipt['passed'] is True and receipt['heldout_used'] is False
            assert receipt['head_sha256'] == result['weight_update_receipt']['after_head_sha256']
        else:
            assert result['status'] in {'failed', 'awaiting_evidence'}
            assert not result.get('saved_base_id')
            assert 'native_admission' not in calls
            assert not store.list('base_candidates')
    finally:
        await service.close()
        store.close()


def test_new_base_cannot_remove_or_rebind_its_distribution_retention_receipt(tmp_path, monkeypatch):
    from lucifer.meditation_retention import CONTRACT, retention_receipt
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    candidate = candidate_fixture(service, store, manager, tmp_path)
    binding = {'head_sha256': candidate['head_sha256'], 'reference_head_sha256': 'b' * 64,
               'retention_contract': CONTRACT, 'training_precision': 'native_f32'}
    candidate = {**candidate, 'candidate_binding': binding, 'id': 'base_' + digest(binding)[:32]}
    try:
        with pytest.raises(ValueError, match='retention receipt'):
            service._verify_base(candidate)
        measured = {'head_sha256': candidate['head_sha256'], 'reference_head_sha256': 'b' * 64,
                    'datasets': {'development': {'tokens': 8, 'forward_kl': {'mean': 0.001, 'maximum': 0.002,
                    'tokens': 8, 'direction': 'untrained_bf16_reference_to_candidate'}}}}
        receipt = retention_receipt(measured, contract=CONTRACT,
            head_sha256=candidate['head_sha256'], reference_sha256='b' * 64)
        candidate['admission_report']['distribution_retention'] = receipt
        service._verify_base(candidate)
        with pytest.raises(ValueError, match='retention binding'):
            service._verify_base({**candidate, 'id': 'base_forged'})
        receipt['head_sha256'] = 'forged'
        with pytest.raises(ValueError, match='another candidate'):
            service._verify_base(candidate)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_resume_finishes_reserved_native_measurement_before_more_training(tmp_path, monkeypatch):
    import lucifer.meditation_sources as source_module
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    calls = mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    numerical = service._worker
    interrupted = {"once": True}
    async def worker(ident, config):
        if config["mode"] == "native_admission" and interrupted["once"]:
            interrupted["once"] = False
            raise RuntimeError("Native measurement response interrupted")
        return await numerical(ident, config)
    async def prepare(*args, **kwargs):
        return bundle()
    monkeypatch.setattr(service, '_worker', worker)
    try:
        await service._run(run['id'], prepare=prepare)
        stopped = store.get('meditation_runs', run['id'])
        assert stopped['status'] == 'failed' and stopped['pending_native_admission']
        pending = stopped['pending_native_admission']
        assert len(stopped['admission_groups_consumed']) == 8
        assert stopped['policy']['admission_attempts'] == 1
        old_calls = list(calls)
        async def forbidden(*args, **kwargs):
            pytest.fail('Reserved native admission must precede source refresh')
        monkeypatch.setattr(source_module, 'prepare_meditation_sources', forbidden)
        await service.resume(run['id'], 'finish-reserved-native-panel')
        await service.tasks[run['id']]
        completed = store.get('meditation_runs', run['id'])
        assert completed['status'] == 'admitted', completed.get('error')
        assert completed['pending_native_admission'] is None
        assert completed['admission_report']['head_sha256'] == pending['config']['head_sha256']
        assert calls == old_calls + ['native_admission']
        assert completed['policy']['successful_updates'] == 25
        assert completed['policy']['admission_attempts'] == 1
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_failed_candidate_load_never_admits_or_activates_saved_artifact(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    manager.failure = 'changed'
    async def prepare(*args, **kwargs):
        return bundle()
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['status'] == 'failed'
        assert result['progress']['restored'] is True
        assert store.get('base_candidates', 'base_fixture')['status'] == 'candidate'
        assert result.get('pending_admission')
        assert not result.get('saved_base_id')
        assert manager.current()['id'] == 'primary'
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_restore_failure_is_visible_and_never_becomes_activatable(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    manager.failure = 'primary'
    async def prepare(*args, **kwargs):
        result = bundle()
        result['examples'] = []
        return result
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['status'] == 'failed'
        assert result['progress']['restored'] is False
        assert 'restoration failed' in result['error']
        assert not store.list('base_candidates')
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_cancel_waits_for_owned_task_restoration_before_returning(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    entered = asyncio.Event()
    async def prepare(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    try:
        service.tasks[run['id']] = asyncio.create_task(service._run(run['id'], prepare=prepare))
        await entered.wait()
        result = await service.cancel(run['id'], 'cancel-one')
        assert result['status'] == 'cancelled'
        assert result['progress']['restored'] is True
        assert not service.busy
        assert manager.current()['id'] == 'primary'
        assert (await service.cancel(run['id'], 'cancel-one'))['status'] == 'cancelled'
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_invalid_end_to_end_response_cannot_admit_a_numerically_improved_head(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    manager.answers_by_model['changed'] = 'wrong response'
    async def prepare(*args, **kwargs):
        return bundle()
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['status'] in {'failed', 'awaiting_evidence'}
        assert not result.get('saved_base_id')
        assert store.get('base_candidates', 'base_fixture')['status'] == 'rejected'
        assert result.get('pending_admission') is None
        assert result['progress']['restored'] is True
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_unscorable_baseline_cannot_make_retention_vacuously_pass(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    manager.answer = 'wrong for every control'
    async def prepare(*args, **kwargs):
        return bundle()
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['status'] in {'failed', 'awaiting_evidence'}
        assert not result.get('saved_base_id')
        assert not any(base.get('status') == 'admitted' for base in store.list('base_candidates'))
        assert result['progress']['restored'] is True
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_resume_reconciles_a_sealed_interrupted_worker_without_repeating_optimizer(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store, status='interrupted')
    output = service._run_dir(run['id']) / 'sealed-worker'
    output.mkdir()
    atomic_json(output / 'result.json', {'mode': 'train', 'successful_updates': 25})
    operation = store.put('meditation_operations', {'id': 'sealed-effect', 'phase': 'worker', 'status': 'prepared',
        'binding': {'run_id': run['id'], 'config': {'mode': 'train', 'output_dir': str(output)}},
        'started_at': 'before-crash'})
    resumed = []
    async def resume_run(ident):
        resumed.append(ident)
    monkeypatch.setattr(service, '_run', resume_run)
    try:
        await service.resume(run['id'], 'reconcile-resume')
        await service.tasks[run['id']]
        committed = store.get('meditation_operations', operation['id'])
        assert committed['status'] == 'completed'
        assert committed['result_sha256'] == file_sha256(output / 'result.json')
        assert resumed == [run['id']]
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_numerical_weight_update_without_native_loading_and_rollback_proof_is_rejected(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    original = service._worker
    async def worker(ident, config):
        if config['mode'] == 'verify':
            return {'passed': True, 'rollback_probe_passed': False}
        return await original(ident, config)
    async def prepare(*args, **kwargs):
        return bundle()
    monkeypatch.setattr(service, '_worker', worker)
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['status'] == 'failed'
        assert 'rollback' in result['error']
        assert not result.get('saved_base_id')
        assert not store.list('base_candidates')
        assert result['progress']['restored'] is True
        assert manager.current()['id'] == 'primary'
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_registering_candidate_profile_does_not_bypass_admission_at_model_load(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    candidate = candidate_fixture(service, store, manager, tmp_path)
    store.update('base_candidates', candidate['id'], status='candidate', rollback_receipt={})
    try:
        with pytest.raises(ValueError, match='passed evaluation'):
            await manager.load('changed')
        assert manager.current()['id'] == 'primary'
        service._evaluation_profile = 'changed'
        await manager.load('changed')
        assert manager.current()['id'] == 'changed'
        service._evaluation_profile = None
        with pytest.raises(ValueError, match='passed evaluation'):
            await manager.load('changed')
    finally:
        service._evaluation_profile = None
        await service.close()
        store.close()


def candidate_sealing_inputs(service, store, manager, tmp_path):
    """A tiny native-format BF16 pack; no inference or optimizer is mocked here."""
    from lucifer.head_pack import load_pack, tensor_hash
    run = seed_run(service, store)
    pack = tmp_path / 'native-pack'
    weights = tmp_path / 'native-weights'
    pack.mkdir()
    weights.mkdir()
    (weights / 'weights.bin').write_bytes(b'\x00\x3f' * 4)
    (pack / 'tensors.txt').write_text('file 0 weights.bin\nt head.weight BF16 0 0 8 2 2\n')
    (pack / 'ds4.txt').write_text(f'model_dir {weights}\nhidden_size 2\nvocab_size 2\n')
    (pack / 'engram.bin').write_bytes(b'fixture-engram')
    (pack / 'tokenizer').mkdir()
    (pack / 'expert-profile.bin').write_bytes(b'fixture-expert-cache')
    parent = manager.specs['primary'].model_copy(update={'path': str(pack)})
    manager.specs[parent.id] = parent
    head = tmp_path / 'learned-head.bf16'
    head.write_bytes(b'\x80\x3f' * 4)
    source = tmp_path / 'sealed-source.json'
    atomic_json(source, bundle())
    runtime = tmp_path / 'native-runtime.json'
    atomic_json(runtime, {'exe': '/bin/true', 'args': ['--pack', str(pack), '--reserve-mib', '1024'],
                          'tokenizer': str(pack / 'tokenizer')})
    native_pack = load_pack(pack)
    exported = {'parent_head_sha256': tensor_hash(native_pack), 'parent_manifest_sha256': native_pack.manifest_sha256}
    report = {'passed': True, 'head_sha256': file_sha256(head), 'source_bundle_sha256': file_sha256(source)}
    trained = {'after_head_sha256': file_sha256(head), 'changed_elements': 4}
    return {'ident': run['id'], 'head_path': str(head), 'exported': exported, 'parent': parent,
            'native': {'runtime_profile': str(runtime)}, 'bundle_path': source, 'report': report, 'trained': trained}


@pytest.mark.asyncio
async def test_sealed_candidate_retry_reuses_immutable_artifacts_after_control_failure(tmp_path, monkeypatch):
    from lucifer.head_pack import load_pack, tensor_hash
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    inputs = candidate_sealing_inputs(service, store, manager, tmp_path)
    try:
        candidate = await service._save_candidate(**inputs)
        saved_head = Path(candidate['artifact_files'][0]['path'])
        original_bytes = saved_head.read_bytes()
        original_mtime = saved_head.stat().st_mtime_ns
        assert '/base-candidates/' + candidate['id'] + '/' in str(saved_head)
        assert candidate['status'] == 'candidate'
        # A failed automatic control evaluation leaves this candidate unadmitted.
        # An explicit retry must inspect/reuse it rather than copy over it.
        def forbidden(*args, **kwargs):
            pytest.fail('Retry attempted to overwrite an existing candidate head')
        monkeypatch.setattr('lucifer.meditation.shutil.copyfileobj', forbidden)
        again = await service._save_candidate(**inputs)
        assert again == candidate
        assert saved_head.read_bytes() == original_bytes and saved_head.stat().st_mtime_ns == original_mtime
        assert len(store.list('base_candidates')) == 1
        profile = json.loads(Path(candidate['spec']['env']['LUCIFER_STRATA_PROFILE']).read_text())
        assert profile['args'][profile['args'].index('--pack') + 1] == candidate['spec']['path']
        assert profile['args'][profile['args'].index('--reserve-mib') + 1] == '2048'
        assert '--profile-readonly' in profile['args']
        assert tensor_hash(load_pack(inputs['parent'].path)) == inputs['exported']['parent_head_sha256']
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_identical_head_different_parent_or_source_lineage_gets_distinct_base_ids(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    inputs = candidate_sealing_inputs(service, store, manager, tmp_path)
    try:
        first = await service._save_candidate(**inputs)
        changed_parent = inputs['parent'].model_copy(update={'description': 'Different pinned parent profile'})
        second = await service._save_candidate(**{**inputs, 'parent': changed_parent})
        new_source = tmp_path / 'different-source.json'
        atomic_json(new_source, {**bundle(), 'additional_source': 'independent-consented-document'})
        source_report = {**inputs['report'], 'source_bundle_sha256': file_sha256(new_source)}
        third = await service._save_candidate(**{**inputs, 'bundle_path': new_source, 'report': source_report})
        assert len({candidate['id'] for candidate in (first, second, third)}) == 3
        assert len({candidate['model_id'] for candidate in (first, second, third)}) == 3
        assert first['head_sha256'] == second['head_sha256'] == third['head_sha256']
        assert len(store.list('base_candidates')) == 3
        for candidate in (first, second, third):
            assert all(file_sha256(Path(entry['path'])) == entry['sha256'] for entry in candidate['artifact_files'])
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_corrupt_existing_candidate_is_rejected_without_repair_or_overwrite(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    inputs = candidate_sealing_inputs(service, store, manager, tmp_path)
    try:
        candidate = await service._save_candidate(**inputs)
        saved_head = Path(candidate['artifact_files'][0]['path'])
        saved_head.write_bytes(b'corrupt!')
        with pytest.raises(ValueError, match='weights, manifest or launcher changed'):
            await service._save_candidate(**inputs)
        assert saved_head.read_bytes() == b'corrupt!'
        assert store.get('base_candidates', candidate['id'])['head_sha256'] == candidate['head_sha256']
        assert Path(inputs['head_path']).read_bytes() != b'corrupt!'
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_later_head_after_failed_controls_preserves_earlier_candidate_bytes(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    inputs = candidate_sealing_inputs(service, store, manager, tmp_path)
    try:
        failed_controls_candidate = await service._save_candidate(**inputs)
        first_head = Path(failed_controls_candidate['artifact_files'][0]['path'])
        before = first_head.read_bytes()
        Path(inputs['head_path']).write_bytes(b'\x00\x40' * 4)
        new_hash = file_sha256(inputs['head_path'])
        later = await service._save_candidate(**{**inputs,
            'report': {**inputs['report'], 'head_sha256': new_hash},
            'trained': {**inputs['trained'], 'after_head_sha256': new_hash}})
        assert later['id'] != failed_controls_candidate['id']
        assert later['model_id'] != failed_controls_candidate['model_id']
        assert first_head.read_bytes() == before
        assert file_sha256(first_head) == failed_controls_candidate['head_sha256']
        assert store.get('base_candidates', failed_controls_candidate['id'])['status'] == 'candidate'
        assert Path(later['artifact_files'][0]['path']).read_bytes() != before
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_candidate_ancestry_uses_actual_parent_and_inherits_ancestor_sources(tmp_path, monkeypatch):
    from lucifer.head_pack import load_pack
    service, store, manager = harness(tmp_path)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    inputs = candidate_sealing_inputs(service, store, manager, tmp_path)
    store.set_setting('active_base_revision', 'unrelated-global-pointer')
    try:
        ancestor = await service._save_candidate(**inputs)
        assert ancestor['parent_base_id'] is None
        ancestor = store.update('base_candidates', ancestor['id'], status='admitted',
                                rollback_receipt={'verified': True})
        parent = manager.specs[ancestor['model_id']]
        child_run = seed_run(service, store, ident='meditation_child')
        store.update('meditation_runs', child_run['id'], base_model_id=parent.id,
                     source_conversation_ids=['child_chat'])
        child_source = tmp_path / 'child-source.json'
        atomic_json(child_source, {**bundle(), 'source_conversation_ids': ['child_chat']})
        child_head = tmp_path / 'child-head.bf16'
        child_head.write_bytes(b'\x00\x40' * 4)
        child_hash = file_sha256(child_head)
        candidate = await service._save_candidate(child_run['id'], str(child_head),
            {'parent_head_sha256': ancestor['head_sha256'],
             'parent_manifest_sha256': load_pack(parent.path).manifest_sha256}, parent,
            {'runtime_profile': parent.env['LUCIFER_STRATA_PROFILE']}, child_source,
            {'passed': True, 'head_sha256': child_hash, 'source_bundle_sha256': file_sha256(child_source)},
            {'after_head_sha256': child_hash, 'changed_elements': 4})
        assert candidate['parent_base_id'] == ancestor['id']
        assert candidate['source_conversation_ids'] == ['chat', 'child_chat']
        assert candidate['candidate_binding']['parent_base_id'] == ancestor['id']
        assert candidate['candidate_binding']['parent_base_identity_sha256']
        assert store.setting('active_base_revision') == 'unrelated-global-pointer'
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_resumed_fresh_sources_preserve_updates_and_reset_development_reference(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store, status='paused')
    run_dir = service._run_dir(run['id'])
    old_bundle = bundle()
    old_revision = service._commit_sources(run['id'], run_dir, old_bundle, [])
    old_source_bytes = Path(old_revision['source_cache_path']).read_bytes()
    old_data_bytes = Path(old_revision['dataset_path']).read_bytes()
    checkpoint = run_dir / 'segment-1' / 'checkpoint'
    checkpoint.mkdir(parents=True)
    previous_head = checkpoint / 'head.bf16'
    previous_head.write_bytes(b'previous-fixture')
    policy = MeditationPolicy(successful_updates=25, segment=1, best_development_loss=0.01, plateau_segments=2)
    store.update('meditation_runs', run['id'], **old_revision, policy=policy.snapshot(),
                 checkpoint_path=str(checkpoint), head_path=str(previous_head),
                 best_checkpoint_path=str(checkpoint), best_head_path=str(previous_head))
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    original = service._worker
    configs = []
    async def worker(ident, config):
        configs.append(config)
        result = await original(ident, config)
        if config['mode'] == 'evaluate' and 'refreshed-development' in config['output_dir']:
            result['datasets']['development']['loss'] = 1.5
        return result
    monkeypatch.setattr(service, '_worker', worker)
    fresh = bundle()
    fresh['examples'].append({**fresh['examples'][0], 'fingerprint': 'new-reviewed-practice'})
    async def prepare(*args, **kwargs):
        return fresh
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['status'] == 'admitted', result.get('error')
        train = next(config for config in configs if config['mode'] == 'train')
        assert train['allow_data_refresh'] is True
        assert train['resume'] == str(checkpoint)
        assert result['policy']['successful_updates'] == 50
        assert result['policy']['segment'] == 2
        assert result['policy']['best_development_loss'] == 1.0
        assert result['policy']['plateau_segments'] == 0
        assert len(result['source_revision_refs']) == 2
        assert result['source_cache_path'] != old_revision['source_cache_path']
        assert Path(old_revision['source_cache_path']).read_bytes() == old_source_bytes
        assert Path(old_revision['dataset_path']).read_bytes() == old_data_bytes
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('crash_phase', ['after_report', 'after_artifact'])
async def test_resume_finishes_first_passing_checkpoint_without_more_training(tmp_path, monkeypatch, crash_phase):
    import lucifer.meditation_sources as source_module
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    calls = mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    save = service._save_candidate
    async def interrupted_save(*args, **kwargs):
        if crash_phase == 'after_artifact':
            await save(*args, **kwargs)
        raise RuntimeError('Simulated interruption after passing admission')
    async def prepare(*args, **kwargs):
        return bundle()
    monkeypatch.setattr(service, '_save_candidate', interrupted_save)
    try:
        await service._run(run['id'], prepare=prepare)
        interrupted = store.get('meditation_runs', run['id'])
        assert interrupted['status'] == 'failed'
        assert interrupted['pending_admission']['binding']['report']['passed'] is True
        assert interrupted['policy']['successful_updates'] == 25
        consumed = list(interrupted['admission_groups_consumed'])
        assert len(consumed) == 8 and len(store.list('meditation_admissions')) == 1
        assert len(store.list('base_candidates')) == (1 if crash_phase == 'after_artifact' else 0)
        worker_calls = list(calls)
        source_path = interrupted['pending_admission']['binding']['source_cache_path']
        source_bytes = Path(source_path).read_bytes()
        async def forbidden(*args, **kwargs):
            pytest.fail('Pending admission replayed training, evaluation or data preparation')
        monkeypatch.setattr(service, '_save_candidate', save)
        monkeypatch.setattr(service, '_worker', forbidden)
        monkeypatch.setattr(source_module, 'prepare_meditation_sources', forbidden)
        if crash_phase == 'after_report':
            manager.selected = None  # Backend restart without model autoload.
        await service.resume(run['id'], 'finish-passing-checkpoint')
        await service.tasks[run['id']]
        completed = store.get('meditation_runs', run['id'])
        assert completed['status'] == 'admitted', completed.get('error')
        assert completed['pending_admission'] is None
        assert completed['saved_base_id'] == 'base_fixture'
        assert completed['policy']['successful_updates'] == 25
        assert completed['admission_groups_consumed'] == consumed
        assert completed['policy']['admission_attempts'] == 1
        assert len(store.list('meditation_admissions')) == 1 and calls == worker_calls
        assert Path(source_path).read_bytes() == source_bytes
        assert manager.current()['id'] == 'primary' and store.setting('active_base_revision') is None
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_resume_after_atomic_admission_before_terminal_status_never_retrains(tmp_path, monkeypatch):
    import lucifer.meditation_sources as source_module
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    update = store.update
    crash = {'enabled': True}
    def interrupted_update(kind, ident, **changes):
        if kind == 'meditation_runs' and changes.get('status') == 'admitted' and crash['enabled']:
            crash['enabled'] = False
            raise RuntimeError('Simulated crash before terminal status')
        return update(kind, ident, **changes)
    async def prepare(*args, **kwargs):
        return bundle()
    monkeypatch.setattr(store, 'update', interrupted_update)
    try:
        with pytest.raises(RuntimeError, match='before terminal status'):
            await service._run(run['id'], prepare=prepare)
        committed = store.get('meditation_runs', run['id'])
        assert committed['saved_base_id'] == 'base_fixture' and committed['pending_admission'] is None
        assert store.get('base_candidates', 'base_fixture')['rollback_receipt']['verified'] is True
        update('meditation_runs', run['id'], status='interrupted')
        async def forbidden(*args, **kwargs):
            pytest.fail('Already admitted base repeated numerical work or source preparation')
        monkeypatch.setattr(service, '_worker', forbidden)
        monkeypatch.setattr(source_module, 'prepare_meditation_sources', forbidden)
        await service.resume(run['id'], 'recover-terminal-status')
        await service.tasks[run['id']]
        result = store.get('meditation_runs', run['id'])
        assert result['status'] == 'admitted'
        assert result['policy']['successful_updates'] == 25
        assert result['progress']['restored'] is True
        assert len(store.list('meditation_admissions')) == 1
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_uncertain_parent_rollback_retains_exact_pending_checkpoint_for_resume(tmp_path, monkeypatch):
    import lucifer.meditation_sources as source_module
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    calls = mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    controls = service._controls
    async def fail_parent_after_candidate(model_id):
        result = await controls(model_id)
        if model_id == 'changed':
            manager.failure = 'primary'
        return result
    async def prepare(*args, **kwargs):
        return bundle()
    monkeypatch.setattr(service, '_controls', fail_parent_after_candidate)
    try:
        await service._run(run['id'], prepare=prepare)
        stopped = store.get('meditation_runs', run['id'])
        assert stopped['status'] == 'failed' and stopped['progress']['restored'] is False
        assert stopped['pending_admission']['phase'] == 'rollback_verification'
        pending_hash = stopped['pending_admission']['binding_sha256']
        assert store.get('base_candidates', 'base_fixture')['status'] == 'candidate'
        old_calls = list(calls)
        manager.failure = None
        monkeypatch.setattr(service, '_controls', controls)
        async def forbidden(*args, **kwargs):
            pytest.fail('Uncertain rollback restarted numerical work or changed the admission panel')
        monkeypatch.setattr(service, '_worker', forbidden)
        monkeypatch.setattr(source_module, 'prepare_meditation_sources', forbidden)
        await service.resume(run['id'], 'recover-parent-rollback')
        await service.tasks[run['id']]
        result = store.get('meditation_runs', run['id'])
        assert result['status'] == 'admitted', result.get('error')
        assert result['pending_admission'] is None
        assert result['policy']['successful_updates'] == 25
        assert result['policy']['admission_attempts'] == 1
        assert calls == old_calls and pending_hash
        assert manager.current()['id'] == 'primary'  # An unadmitted leftover is never the restore target.
        assert result['rollback_receipt']['control_results_reproduced'] is True
    finally:
        await service.close()
        store.close()


def native_admission_fixture(service, store, run, tmp_path, monkeypatch):
    source = tmp_path / 'admission-sources.json'
    atomic_json(source, {})
    native = {'engine': '/bin/true'}
    monkeypatch.setattr(service, '_native_config', lambda spec: native)
    monkeypatch.setattr(service, '_check_sources', lambda value: None)
    store.update('meditation_runs', run['id'], source_cache_path=str(source), source_bundle_sha256=file_sha256(source),
                 native_runtime_binding={'config': native}, dataset_path=str(source), dataset_sha256=file_sha256(source))


@pytest.mark.asyncio
async def test_late_admission_attempt_grows_group_count_without_relaxing_alpha(tmp_path, monkeypatch):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    head = tmp_path / 'candidate.bin'
    head.write_bytes(b'fixture')
    native_admission_fixture(service, store, run, tmp_path, monkeypatch)
    policy = MeditationPolicy(admission_attempts=4)
    groups = [f'fresh-{index}' for index in range(12)]
    prepared = {'datasets': {'holdout': {'groups': groups,
        'native_group_clipped_losses': {group: 2.0 for group in groups},
        'original_native_group_clipped_losses': {group: 1.98 for group in groups}}}}
    baseline = {'datasets': {'holdout': {'group_clipped_losses': {group: 2.0 for group in groups}}}}
    async def measured(ident, config):
        assert config['mode'] == 'native_admission'
        selected = config['selected_groups']
        return {'backend': 'native_engine_teacher_forced', 'head_sha256': config['head_sha256'],
                'group_clipped_losses': {group: 1.0 for group in selected},
                'converted_group_clipped_losses': {group: 2.0 for group in selected},
                'original_group_clipped_losses': {group: 1.98 for group in selected}}
    monkeypatch.setattr(service, '_worker', measured)
    monkeypatch.setattr(service, '_admission_snapshot', lambda *args: {'fixture_snapshot': True})
    try:
        consumed = set()
        report = await service._admission(run['id'], str(head), {'shape': [1, 1]}, prepared,
            baseline, policy, consumed, service._run_dir(run['id']))
        assert report['passed'] is True
        assert report['alpha'] == 0.05 / (5 * 6)
        assert len(report['groups']) == len(consumed) == 10
        assert report['converted_baseline']['p_value'] <= report['alpha']
        assert policy.admission_attempts == 5
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_late_admission_with_too_few_groups_does_not_spend_error_budget(tmp_path):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    policy = MeditationPolicy(admission_attempts=4)
    try:
        consumed = set()
        report = await service._admission(run['id'], 'unused', {},
            {'datasets': {'holdout': {'groups': [str(i) for i in range(9)]}}}, {}, policy, consumed,
            service._run_dir(run['id']))
        assert report['status'] == 'awaiting_evidence' and report['required_groups'] == 10
        assert policy.admission_attempts == 4 and consumed == set()
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_admission_commit_is_atomic_when_run_update_fails(tmp_path, monkeypatch):
    import lucifer.meditation_sources as source_module
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    calls = mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    update = store.update
    once = {'fail': True}
    def fail_commit(kind, ident, **changes):
        if kind == 'meditation_runs' and changes.get('saved_base_id') and once['fail']:
            once['fail'] = False
            raise RuntimeError('Interrupted atomic admission commit')
        return update(kind, ident, **changes)
    async def prepare(*args, **kwargs):
        return bundle()
    monkeypatch.setattr(store, 'update', fail_commit)
    try:
        await service._run(run['id'], prepare=prepare)
        failed = store.get('meditation_runs', run['id'])
        assert failed['status'] == 'failed' and failed['pending_admission']
        assert failed.get('saved_base_id') is None
        assert store.get('base_candidates', 'base_fixture')['status'] == 'candidate'
        assert store.list('meditation_admissions')[0]['status'] == 'pending_finalization'
        old_calls = list(calls)
        async def forbidden(*args, **kwargs):
            pytest.fail('Atomic commit recovery repeated numerical work')
        monkeypatch.setattr(service, '_worker', forbidden)
        monkeypatch.setattr(source_module, 'prepare_meditation_sources', forbidden)
        await service.resume(run['id'], 'retry-admission-commit')
        await service.tasks[run['id']]
        recovered = store.get('meditation_runs', run['id'])
        assert recovered['status'] == 'admitted' and recovered['pending_admission'] is None
        assert store.get('base_candidates', 'base_fixture')['status'] == 'admitted'
        assert calls == old_calls and recovered['policy']['successful_updates'] == 25
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('original_loss', [0.99, None])
async def test_bf16_improvement_cannot_hide_regression_or_missing_original_parent_scores(tmp_path, monkeypatch, original_loss):
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    head = tmp_path / 'candidate.bf16'
    head.write_bytes(b'fixture')
    native_admission_fixture(service, store, run, tmp_path, monkeypatch)
    groups = [f'independent-{index}' for index in range(8)]
    heldout = {'groups': groups, 'native_group_clipped_losses': {group: 1.01 for group in groups}}
    if original_loss is not None:
        heldout['original_native_group_clipped_losses'] = {group: original_loss for group in groups}
    prepared = {'datasets': {'holdout': heldout}}
    baseline = {'datasets': {'holdout': {'group_clipped_losses': {group: 1.01 for group in groups}}}}
    async def measured(ident, config):
        assert config['mode'] == 'native_admission'
        return {'backend': 'native_engine_teacher_forced', 'head_sha256': config['head_sha256'],
                'group_clipped_losses': {group: 1.0 for group in groups},
                'converted_group_clipped_losses': {group: 1.01 for group in groups},
                'original_group_clipped_losses': {group: original_loss for group in groups} if original_loss is not None else {}}
    def no_passing_snapshot(*args, **kwargs):
        pytest.fail('A BF16-only improvement reached the passing-checkpoint journal')
    monkeypatch.setattr(service, '_worker', measured)
    monkeypatch.setattr(service, '_admission_snapshot', no_passing_snapshot)
    try:
        report = await service._admission(run['id'], str(head), {'shape': [1, 1]}, prepared,
            baseline, MeditationPolicy(), set(), service._run_dir(run['id']))
        assert report['converted_baseline']['passed'] is True
        assert report['original_baseline']['passed'] is False
        assert report['passed'] is False
        assert store.get('meditation_runs', run['id']).get('pending_admission') is None
        assert not store.list('base_candidates')
        if original_loss is None:
            assert report['original_baseline']['status'] == 'awaiting_evidence'
        else:
            assert report['original_baseline']['mean_clipped_loss_improvement'] < 0
    finally:
        await service.close()
        store.close()
