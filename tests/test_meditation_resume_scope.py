import base64
import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lucifer.adaptive import atomic_json, file_sha256
from lucifer.chat import ChatService
from lucifer.meditation_sources import prepare_meditation_sources
from lucifer.schemas import MeditationResume
from test_meditation_service import harness, seed_run


async def source(store, name, *, text=None):
    chat = ChatService(store, SimpleNamespace())
    cid = chat.create(title=name, training_enabled=True)['id']
    document = await chat.knowledge.attach(cid, filename=name + '.txt',
        content_base64=base64.b64encode((text or 'Distinct source passage about ' + name).encode()).decode())
    chat.knowledge.read(cid, document['id'], 0, 5)
    return cid


async def stopped(service, store):
    ids = [await source(store, name) for name in ['training', 'development', 'heldout']]
    roles = dict(zip(ids, ['train', 'development', 'heldout']))
    request = {'mode': 'supervised', 'conversation_ids': ids, 'source_roles': roles, 'idempotency_key': 'original'}
    bundle = await prepare_meditation_sources(store, request)
    run = seed_run(service, store, status='awaiting_evidence')
    path = service._run_dir(run['id']) / 'old-sources.json'
    atomic_json(path, bundle)
    checkpoint = service._run_dir(run['id']) / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'rng').write_bytes(b'exact-original-state')
    consumed = [g['id'] for g in bundle['manifest']['groups'] if g['partition'] == 'heldout']
    store.update('meditation_runs', run['id'], request=request, source_cache_path=str(path),
        source_bundle_sha256=file_sha256(path), source_manifest=bundle['manifest'], checkpoint_path=str(checkpoint),
        admission_groups_consumed=consumed)
    return store.get('meditation_runs', run['id']), ids


@pytest.mark.asyncio
async def test_explicit_append_retains_checkpoint_consumed_groups_and_retry_identity(tmp_path, monkeypatch):
    service, store, _ = harness(tmp_path)
    run, ids = await stopped(service, store)
    fresh = await source(store, 'fresh-heldout')
    before = copy.deepcopy(run)
    calls = []
    async def continuation(ident):
        calls.append(ident)
    monkeypatch.setattr(service, '_run', continuation)
    try:
        result = await service.resume(run['id'], 'expand', conversation_ids=[*ids, fresh], source_roles={fresh: 'heldout'})
        await service.tasks[run['id']]
        assert result['checkpoint_path'] == before['checkpoint_path']
        assert result['admission_groups_consumed'] == before['admission_groups_consumed']
        assert result['request']['conversation_ids'] == [*ids, fresh]
        assert result['request']['source_roles'] == {**before['request']['source_roles'], fresh: 'heldout'}
        assert len(result['source_scope_history']) == 1
        # A validation preview cannot reserve a fresh group before the actual
        # continuation owns source preparation.
        assert len(store.list('meditation_source_partitions')) == 3
        assert (tmp_path / 'state/meditation' / run['id'] / 'checkpoint/rng').read_bytes() == b'exact-original-state'
        again = await service.resume(run['id'], 'expand', conversation_ids=[*ids, fresh], source_roles={fresh: 'heldout'})
        assert again == result and calls == [run['id']]
        with pytest.raises(ValueError, match='different Meditation request'):
            await service.resume(run['id'], 'expand', conversation_ids=ids)
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['narrow', 'missing', 'consent', 'repartition', 'pending'])
async def test_invalid_extension_is_rejected_before_request_or_checkpoint_mutation(tmp_path, monkeypatch, fault):
    service, store, _ = harness(tmp_path)
    run, ids = await stopped(service, store)
    fresh = await source(store, 'fresh')
    options = {'conversation_ids': [*ids, fresh]}
    if fault == 'narrow':
        options['conversation_ids'] = [ids[0], fresh]
    elif fault == 'missing':
        options['conversation_ids'].append('absent')
    elif fault == 'consent':
        store.update('conversations', fresh, training_enabled=False)
    elif fault == 'repartition':
        options['source_roles'] = {ids[0]: 'heldout'}
    else:
        store.update('meditation_runs', run['id'], pending_native_admission={'identity': 'already-reserved'})
    async def forbidden(*args, **kwargs):
        pytest.fail('Invalid scope launched numerical continuation')
    monkeypatch.setattr(service, '_run', forbidden)
    try:
        with pytest.raises(ValueError):
            await service.resume(run['id'], 'bad-' + fault, **options)
        current = store.get('meditation_runs', run['id'])
        assert current['request'] == run['request']
        assert current['checkpoint_path'] == run['checkpoint_path']
        assert current['admission_groups_consumed'] == run['admission_groups_consumed']
        assert not current.get('source_scope_history') and not service.tasks
        assert len(store.list('meditation_source_partitions')) == 3
    finally:
        await service.close()
        store.close()


def test_resume_schema_does_not_accept_training_parameter_changes():
    for scope in ([], ['same', 'same'], [' ']):
        with pytest.raises(ValidationError):
            MeditationResume(idempotency_key='scope', conversation_ids=scope)
    with pytest.raises(ValidationError):
        MeditationResume(idempotency_key='scope', mode='unsupervised')
    with pytest.raises(ValidationError):
        MeditationResume(idempotency_key='scope', source_roles={'chat': 'easy-test'})
    assert json.loads(MeditationResume(idempotency_key='legacy').model_dump_json(exclude_none=True)) == {'idempotency_key': 'legacy'}
