"""Controller wiring checks, distinct from the actual native pilot receipt."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lucifer.behavioral_job import prepare_resources
from lucifer.schemas import MeditationRequest, SleepRequest
from lucifer.sleep_behavioral import evaluate_outcomes
from test_meditation_service import harness, seed_run, bundle, mocked_numerical_run


def test_both_new_api_modes_default_to_outcomes_and_pilot_is_bounded():
    assert SleepRequest().evaluation_protocol == 'behavioral_v1'
    assert MeditationRequest(idempotency_key='new').evaluation_protocol == 'behavioral_v1'
    for bad in (0, 101, True):
        with pytest.raises(ValueError):
            MeditationRequest(idempotency_key='new', pilot_updates=bad)


def test_automatic_validation_resources_support_both_head_training_encodings():
    from lucifer.behavioral_resources import attach_numeric_validation
    from lucifer.meditation_sources import _worker_dataset
    for mode in ('supervised', 'unsupervised'):
        value = bundle()
        value['development_examples'], value['holdout_examples'] = [], []
        value['manifest']['training_mode'] = mode
        if mode == 'unsupervised':
            value['examples'][0].pop('messages')
            value['examples'][0]['text'] = 'A consented source passage.'
        result = attach_numeric_validation(value, 'private-independent-seed')
        datasets = _worker_dataset(result)['datasets']
        assert len(datasets['development']) == len(datasets['holdout']) == 2
        assert all(('text' in row) == (mode == 'unsupervised') for key in datasets for row in datasets[key])
        assert datasets['train'] == _worker_dataset(value)['datasets']['train']


@pytest.mark.asyncio
async def test_small_meditation_routes_real_receipts_to_outcomes_then_restores(tmp_path, monkeypatch):
    import lucifer.meditation_behavioral as outcomes
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    store.update('meditation_runs', run['id'], request={**run['request'], 'evaluation_protocol': 'behavioral_v1', 'pilot_updates': 1})
    calls = mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    async def prepare(*args, **kwargs):
        return bundle()
    async def reply(*args, **kwargs):
        return {'complete': True, 'answer': '{}'}
    service.learning.manager = manager
    service.learning._evaluation_token_count = reply
    service.learning._evaluation_answer = reply
    seen = []
    async def admission(svc, ident, head, exported, directory, *, numerical_retention):
        job = store.get('meditation_runs', ident)
        assert numerical_retention and job['native_update_receipt']['rollback_probe_passed']
        assert job['weight_update_receipt']['segment_updates'] == 1
        assert Path(job['behavioral_resources']['path']).is_file()
        assert job['teaching_plan']['master_prompt']['mode'] == 'meditation'
        seen.append(head)
        return {'passed': False, 'status': 'rejected'}
    monkeypatch.setattr(outcomes, 'admission', admission)
    try:
        await service._run(run['id'], prepare=prepare)
        final = store.get('meditation_runs', run['id'])
        assert final['status'] == 'pilot_complete', final.get('error')
        assert seen and calls.count('train') == 4  # 15 calibration + 1 retained update.
        assert final['policy']['successful_updates'] == 1
        assert final['retained_checkpoint_updates'] == 1
        assert final['progress']['restored'] and manager.current()['id'] == 'primary'
        assert not final['resumable'] and not store.setting('active_base_revision')
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_sleep_outcomes_are_default_resources_owned_effects_and_restore_adapter(tmp_path):
    from lucifer.config import Settings
    from lucifer.store import Store
    store = Store(tmp_path / 'state.sqlite')
    settings = Settings(data_dir=tmp_path / 'runtime', models_file=tmp_path / 'no-models.yaml')
    restored = []
    class Manager:
        def current(self):
            return {'id': 'student', 'backend': 'external'}
        async def activate_adapter(self, adapter):
            restored.append(adapter)
    async def answer(*args):
        return {'answer': '{}', 'complete': True}
    async def count(*args):
        return None
    service = SimpleNamespace(store=store, settings=settings, manager=Manager(),
        _require_source_consent=lambda adapter: None, _kind=lambda ident: 'sleep_runs',
        _evaluation_answer=answer, _evaluation_token_count=count, _progress=lambda *a, **k: None)
    store.put('sleep_runs', {'id': 'sleep-test', 'request': {'evaluation_protocol': 'behavioral_v1'}})
    adapter = store.put('adapters', {'id': 'candidate', 'training_run_id': 'sleep-test', 'base_model_id': 'student',
                                    'checkpoint_manifest_sha256': 'a' * 64})
    try:
        report = await evaluate_outcomes(service, adapter)
        assert report['complete'] and not report['passed']
        assert len(report['categories']) == 4 and report['total_tasks'] == 8
        assert restored == [None]
        job = store.get('sleep_runs', 'sleep-test')
        assert job['behavioral_report'] == report
        selected = json.loads((Path(job['behavioral_resources']['path']).parent.parent / 'candidate/selection.json').read_text())
        assert selected['adapter_artifact_sha256'] == 'a' * 64
        # Completed measurements are reused without another inference/tool call.
        async def forbidden(*args):
            pytest.fail('Completed outcome work was repeated')
        service._evaluation_answer = forbidden
        assert await evaluate_outcomes(service, adapter) == report
        assert len(store.get('sleep_runs', 'sleep-test')['behavioral_attempts']) == 1
    finally:
        store.close()


def test_repeated_checkpoint_search_tightens_the_run_confidence_budget():
    from lucifer.sleep_behavioral import sequential_report
    first = sequential_report({'passed': True, 'complete': True, 'paired_test': {'p_value': 1 / 64}}, 1)
    second = sequential_report({'passed': True, 'complete': True, 'paired_test': {'p_value': 1 / 64}}, 2)
    assert first['passed'] and not second['passed']
    assert second['sequential_gate']['alpha'] < first['sequential_gate']['alpha']


@pytest.mark.asyncio
@pytest.mark.parametrize('safe', [True, False])
async def test_adaptive_sleep_outcomes_cannot_bypass_existing_safety_controls(tmp_path, monkeypatch, safe):
    from lucifer import adaptive_learning as controller
    from lucifer.store import Store
    import lucifer.sleep_behavioral as behavioral
    store = Store(tmp_path / 'state.sqlite')
    store.put('sleep_runs', {'id': 'sleep_safety', 'request': {'evaluation_protocol': 'behavioral_v1'}})
    adapter = store.put('adapters', {'id': 'adapter', 'training_run_id': 'sleep_safety', 'base_model_id': 'student',
        'checkpoint_manifest_sha256': 'sealed-artifact', 'evaluation_suite': {}, 'metrics': {}})
    suite = {'cases': [{'id': 'privacy', 'suite': 'safety'}, {'id': 'json', 'suite': 'format'}]}
    restored = []
    async def activate(value):
        restored.append(value)
    service = SimpleNamespace(store=store, _require_source_consent=lambda value: None,
        _adapter_suite=lambda value: (suite, suite['cases'], 'sealed-suite'),
        manager=SimpleNamespace(activate_adapter=activate))
    async def outcomes(*args):
        return {'passed': True, 'complete': True, 'task_success_delta': .5}
    async def anchors(*args, **kwargs):
        return [{'id': 'privacy', 'suite': 'safety', 'scored': True, 'baseline_pass': True,
                 'candidate_pass': safe, 'baseline_latency': 1., 'candidate_latency': 1.}]
    monkeypatch.setattr(behavioral, 'evaluate_outcomes', outcomes)
    monkeypatch.setattr(controller, 'measure_panel', anchors)
    try:
        result = await controller.evaluate_checkpoint(service, 'sleep_safety', {'baseline_binding': {}}, adapter)
        assert result['passed'] is safe
        assert result['gates']['fixed_safety'] is safe and result['gates']['fixed_regression'] is safe
        assert restored == [None]
        assert (await controller.evaluate_checkpoint(service, 'sleep_safety', {'baseline_binding': {}}, adapter)) == result
    finally:
        store.close()


@pytest.mark.asyncio
async def test_planner_checks_loaded_model_proposals_without_exposing_evaluation(tmp_path):
    from lucifer.training_planner import plan_meditation
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    values = bundle()
    values['examples'][0]['source_authority'] = 'user_correction'
    seen = []
    async def count(*args):
        return None
    async def reply(messages, *args):
        seen.append(messages)
        return {'complete': True, 'answer': '{"priority":["user_correction"]}'}
    service.learning._evaluation_token_count, service.learning._evaluation_answer = count, reply
    try:
        result = await plan_meditation(service, run['id'], values)
        assert result['examples'] == values['examples']
        plan = store.get('meditation_runs', run['id'])['teaching_plan']
        assert plan['selection'] == 'loaded_model_checked_enum_proposal'
        assert 'LUCIFER_TRAINING_MASTER_V1' in seen[0][0]['content']
        assert 'oracle' not in seen[0][1]['content']
        await plan_meditation(service, run['id'], values)
        assert len(seen) == 1
        resources = prepare_resources(service._run_dir(run['id']) / 'test-resources', mode='meditation')
        assert resources['master_prompt'] == plan['master_prompt']
    finally:
        store.close()


@pytest.mark.asyncio
async def test_native_outcome_path_rejects_no_gain_and_removes_owned_parent_profile(tmp_path, monkeypatch):
    from lucifer.head_capture import controlled_execution_plan
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    store.update('meditation_runs', run['id'], request={**run['request'], 'evaluation_protocol': 'behavioral_v1', 'pilot_updates': 1})
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    original_save = service._save_candidate
    async def save(*args):
        result = await original_save(*args)
        return store.update('base_candidates', result['id'], execution_plan=controlled_execution_plan({}))
    service._save_candidate = save
    service.learning.manager = manager
    async def count(*args):
        return None
    async def reply(*args):
        return {'complete': True, 'answer': '{}'}
    service.learning._evaluation_token_count, service.learning._evaluation_answer = count, reply
    async def prepare(*args, **kwargs):
        return bundle()
    try:
        await service._run(run['id'], prepare=prepare)
        result = store.get('meditation_runs', run['id'])
        assert result['status'] == 'pilot_complete', result.get('error')
        assert result['behavioral_report']['complete'] and not result['behavioral_report']['passed']
        assert len(result['behavioral_report']['categories']) == 4
        assert manager.current()['id'] == 'primary'
        assert run['id'] + '-parent-evaluation' not in manager.specs
        assert store.get('base_candidates', 'base_fixture')['status'] == 'rejected'
        assert service._evaluation_profile is None
        with pytest.raises(ValueError, match='passed evaluation'):
            await service.activate('base_fixture', 'cannot-activate')
    finally:
        await service.close()
        store.close()


def test_source_withdrawal_purges_outcome_resources_and_late_owned_writes(tmp_path):
    from lucifer.chat import ChatService
    from lucifer.store import Store
    from lucifer.sleep_behavioral import purge_revoked_resources
    store = Store(tmp_path / 'state.sqlite')
    chat = ChatService(store, None)
    cid = chat.create(training_enabled=True)['id']
    root = tmp_path / 'evaluations/sleep_privacy'
    root.mkdir(parents=True)
    (root / 'private-episode.json').write_text('{}')
    store.put('sleep_runs', {'id': 'sleep_privacy', 'status': 'running', 'source_conversation_ids': [cid]})
    try:
        chat.patch(cid, {'training_enabled': False})
        assert not root.exists()
        root.mkdir()
        (root / 'late-result.json').write_text('{}')
        purge_revoked_resources(store, 'sleep_privacy')
        assert not root.exists()
        outside = tmp_path / 'unrelated'
        outside.mkdir()
        (outside / 'keep').write_text('unrelated')
        root.symlink_to(outside, target_is_directory=True)
        purge_revoked_resources(store, 'sleep_privacy')
        assert not root.is_symlink() and (outside / 'keep').read_text() == 'unrelated'
    finally:
        store.close()


@pytest.mark.asyncio
async def test_standard_meditation_retains_adaptive_search_with_fresh_outcome_draws(tmp_path, monkeypatch):
    import lucifer.meditation_behavioral as outcomes
    service, store, manager = harness(tmp_path)
    run = seed_run(service, store)
    store.update('meditation_runs', run['id'], request={**run['request'], 'evaluation_protocol': 'behavioral_v1'})
    mocked_numerical_run(service, store, manager, tmp_path, monkeypatch)
    service.learning.manager = manager
    async def reply(*args, **kwargs):
        return {'complete': True, 'answer': '{}'}
    service.learning._evaluation_token_count = reply
    service.learning._evaluation_answer = reply
    suites = []
    async def admission(svc, ident, *args, **kwargs):
        suites.append(store.get('meditation_runs', ident)['behavioral_resources']['suite_sha256'])
        return {'passed': False}
    monkeypatch.setattr(outcomes, 'admission', admission)
    async def prepare(*args, **kwargs):
        return bundle()
    try:
        await service._run(run['id'], prepare=prepare)
        final = store.get('meditation_runs', run['id'])
        assert final['status'] == 'rejected', final.get('error')
        assert len(suites) == len(set(suites)) == 4
        assert final['policy']['plateau_segments'] == 3
        assert final['progress']['restored'] and manager.current()['id'] == 'primary'
    finally:
        await service.close()
        store.close()
