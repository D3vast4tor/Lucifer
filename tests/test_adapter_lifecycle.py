import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from lucifer.adapter_lifecycle import activate_expert, delete_adapter
from lucifer.adaptive_learning import gate_report
from lucifer.store import Store


@pytest.fixture
def service(tmp_path):
    store = Store(tmp_path / 'state.db')
    selected = {'id': 'primary', 'adapter_id': None}
    async def load(ident):
        selected['id'] = ident
    async def activate(adapter):
        selected['adapter_id'] = adapter['id'] if adapter else None
    manager = SimpleNamespace(current=lambda: selected, load=load, activate_adapter=activate)
    value = SimpleNamespace(store=store, manager=manager, settings=SimpleNamespace(data_dir=tmp_path),
                            lock=asyncio.Lock(), busy=False)
    yield value
    store.close()


def adapter(service, ident='unused'):
    path = service.settings.data_dir / 'adapters' / ident
    path.mkdir(parents=True)
    (path / 'weights').write_bytes(b'checkpoint')
    return service.store.put('adapters', {'id': ident, 'path': str(path), 'status': 'failed'})


def test_trash_is_idempotent_and_preserves_provenance(service):
    row = adapter(service)
    result = delete_adapter(service, row['id'])
    assert not Path(row['path']).exists()
    assert (Path(result['trash_path']) / 'weights').read_bytes() == b'checkpoint'
    assert result['status'] == 'deleted'
    with pytest.raises(KeyError):
        service.store.get('adapters', row['id'])
    assert delete_adapter(service, row['id']) == result


@pytest.mark.parametrize('protection', ['active', 'descendant', 'resumable', 'shared'])
def test_protected_experts_are_not_moved(service, protection):
    row = adapter(service)
    if protection == 'active':
        service.store.set_setting('active_adapter', row['id'])
    elif protection == 'descendant':
        service.store.put('adapters', {'id': 'child', 'source_expert_ids': [row['id']]})
    elif protection == 'shared':
        service.store.put('adapters', {'id': 'shared', 'path': row['path']})
    else:
        service.store.put('sleep_runs', {'id': 'sleep_resume', 'status': 'cancelled', 'resumable': True,
                                        'candidate_adapter_id': row['id']})
    with pytest.raises(ValueError):
        delete_adapter(service, row['id'])
    assert Path(row['path']).exists() and not service.store.list('deleted_adapters')


def test_external_weights_are_never_removed(service, tmp_path):
    external = tmp_path / 'external'
    external.mkdir()
    service.store.put('adapters', {'id': 'external', 'path': str(external)})
    result = delete_adapter(service, 'external')
    assert result['external_artifact_preserved'] and external.exists()


@pytest.mark.asyncio
async def test_explicit_activation_switches_matching_profile_and_restores_on_error(service, monkeypatch):
    service.store.put('adapters', {'id': 'expert', 'admitted_at': 'today', 'base_model_id': 'compact'})
    monkeypatch.setattr('lucifer.adaptive_learning.validate_admission', lambda *args: None)
    async def admitted(locked_service, ident):
        assert locked_service is service and service.lock.locked()
        assert service.manager.current()['id'] == 'compact'
        return {'id': ident}
    monkeypatch.setattr('lucifer.adaptive_learning._activate_admitted', admitted)
    assert await activate_expert(service, 'expert') == {'id': 'expert'}
    await service.manager.load('primary')
    async def fail(locked_service, ident):
        raise ValueError('source changed')
    monkeypatch.setattr('lucifer.adaptive_learning._activate_admitted', fail)
    with pytest.raises(ValueError):
        await activate_expert(service, 'expert')
    assert service.manager.current()['id'] == 'primary'


@pytest.mark.asyncio
async def test_loading_expert_during_sleep_preserves_the_loaded_model(service, monkeypatch):
    service.store.put('adapters', {'id': 'expert', 'admitted_at': 'today', 'base_model_id': 'compact'})
    service.busy = True
    monkeypatch.setattr('lucifer.adaptive_learning.validate_admission', lambda *args: None)
    loads = []
    async def load(ident):
        loads.append(ident)
    service.manager.load = load
    with pytest.raises(ValueError, match='Sleep to finish'):
        await activate_expert(service, 'expert')
    assert loads == []
    assert service.manager.current()['id'] == 'primary'
    assert service.store.list('audit') == []


@pytest.mark.asyncio
async def test_model_switch_holds_learning_lock_until_expert_activation(service, monkeypatch):
    service.store.put('adapters', {'id': 'expert', 'admitted_at': 'today', 'base_model_id': 'compact'})
    monkeypatch.setattr('lucifer.adaptive_learning.validate_admission', lambda *args: None)
    loading, finish_loading = asyncio.Event(), asyncio.Event()
    original_load = service.manager.load
    async def load(ident):
        loading.set()
        await finish_loading.wait()
        await original_load(ident)
    service.manager.load = load
    async def admitted(locked_service, ident):
        assert locked_service.lock.locked()
        return {'id': ident}
    monkeypatch.setattr('lucifer.adaptive_learning._activate_admitted', admitted)
    activation = asyncio.create_task(activate_expert(service, 'expert'))
    await loading.wait()
    assert service.lock.locked()
    waiter = asyncio.create_task(service.lock.acquire())
    await asyncio.sleep(0)
    assert not waiter.done()
    finish_loading.set()
    assert await activation == {'id': 'expert'}
    await waiter
    service.lock.release()


def test_regression_is_not_constant_zero_and_missing_evidence_cannot_pass():
    suite = {'cases': [{'id': str(i)} for i in range(21)], 'metadata': {'promotion_eligible': True}}
    rows = [{'id': str(i), 'suite': 'core', 'repetition': r, 'baseline_pass': i < 9,
             'candidate_pass': i < 12, 'scored': True, 'baseline_latency': 1., 'candidate_latency': 1.}
            for i in range(21) for r in range(3)]
    baseline = copy.deepcopy(rows)
    metrics, gates = gate_report(suite, copy.deepcopy(rows), pinned_baseline=baseline)
    assert metrics['regression_rate'] == 0 and metrics['baseline_passing_pairs'] == 27
    broken = copy.deepcopy(rows)
    broken[0]['candidate_pass'] = False
    metrics, gates = gate_report(suite, broken, pinned_baseline=baseline)
    assert metrics['regression_rate'] == pytest.approx(1 / 63) and not gates['regression']
    assert metrics['regression_rate_on_baseline_passes'] == pytest.approx(1 / 27)
    _, gates = gate_report(suite, rows[:-3], pinned_baseline=baseline[:-3])
    assert not gates['complete_evidence']


@pytest.mark.asyncio
async def test_cortex_activation_is_explicit_and_delete_requires_confirmation(monkeypatch):
    import httpx
    from lucifer.client import APIClient
    from lucifer.tui import LuciferApp, DeleteExpertScreen
    from lucifer.adapter_map import AdapterBrain
    from test_sleep_counters import Heartbeat
    backend = Heartbeat({'id': 'sleep_done', 'status': 'admitted', 'progress': {'terminal': True, 'restored': True}}, busy=False)
    backend.snapshot['models'].append({'id': 'compact', 'backend': 'vllm'})
    backend.snapshot['adapters'] = [{'id': 'expert', 'status': 'admitted', 'admitted_at': 'today', 'base_model_id': 'compact'}]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    calls = []
    async with app.run_test(size=(150, 60)) as pilot:
        monkeypatch.setattr(app, 'submit', calls.append)
        app.set_view('brain')
        brain = app.query_one('#brain-canvas', AdapterBrain)
        brain.selected_id = 'adapter:expert'
        await app.brain_selected(AdapterBrain.Selected(brain.selected()))
        assert 'LOAD BASE & ACTIVATE' in str(app.query_one('#brain-select').label)
        app.brain_activate()
        assert calls == ['/adapter activate expert']
        app.brain_delete()
        await pilot.pause()
        assert isinstance(app.screen, DeleteExpertScreen)
        await pilot.click('#keep-expert')
        await pilot.pause()
        assert calls == ['/adapter activate expert']
        app.brain_delete()
        await pilot.pause()
        await pilot.click('#confirm-delete-expert')
        await pilot.pause()
        assert calls[-1] == '/adapter delete expert'
