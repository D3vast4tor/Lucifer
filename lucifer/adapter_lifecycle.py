"""Explicit expert activation and reversible removal from the registry."""
from pathlib import Path

from .store import now, uid


async def activate_expert(service, ident):
    # Loading a different base is itself a model mutation. Serialize it with
    # Sleep planning and reject busy jobs before replacing their teacher.
    async with service.lock:
        if service.busy:
            raise ValueError('Wait for Sleep to finish before loading an expert')
        return await _activate_expert(service, ident)


async def _activate_expert(service, ident):
    adapter = service.store.get('adapters', ident)
    if not adapter.get('admitted_at'):
        raise ValueError('This expert has not passed admission')
    from .adaptive_learning import _activate_admitted, validate_admission
    validate_admission(service, adapter)
    previous = dict(service.manager.current() or {})
    active = service.store.setting('active_adapter')
    if active == ident and previous.get('adapter_id') == ident and previous.get('id') == adapter['base_model_id']:
        return adapter
    service.store.audit('expert_activation_requested', adapter_id=ident, base_model_id=adapter['base_model_id'])
    try:
        await service.manager.load(adapter['base_model_id'])
        return await _activate_admitted(service, ident)
    except BaseException:
        if previous:
            await service.manager.load(previous['id'])
            old = service.store.get('adapters', active) if active else None
            await service.manager.activate_adapter(old)
        raise


def delete_adapter(service, ident):
    """Move only owned, unshared artifacts to Trash; retain an audit tombstone.

    Adapter IDs remain reserved after deletion. Repeating DELETE reuses the
    prepared move receipt, including recovery after rename but before commit.
    """
    store = service.store
    try:
        old = store.get('deleted_adapters', ident)
    except KeyError:
        old = None
    if old and old.get('status') == 'deleted':
        return old
    adapter = store.get('adapters', ident)
    current = service.manager.current() or {}
    if ident == store.setting('active_adapter') or current.get('adapter_id') == ident or adapter.get('status') == 'active':
        raise ValueError('Select the base model or another expert before deleting the active expert')
    for other in store.list('adapters'):
        if other['id'] == ident:
            continue
        parents = set(other.get('source_expert_ids') or []) | {other.get('parent_adapter_id'), other.get('base_adapter_id')}
        if ident in parents:
            raise ValueError('Another expert depends on this adapter; remove its descendants first')
    from .learning import _TERMINAL
    path = Path(adapter['path']).expanduser() if adapter.get('path') else None
    for kind in ('sleep_runs', 'training_runs'):
        for job in store.list(kind):
            if job.get('status') in _TERMINAL and not job.get('resumable'):
                continue
            adaptive, request = job.get('adaptive') or {}, job.get('request') or {}
            refs = {job.get('candidate_adapter_id'), request.get('base_adapter'),
                    adaptive.get('previous_round_adapter_id'), adaptive.get('baseline_adapter_id')}
            refs.update(request.get('source_expert_ids') or [])
            checkpoint = (adaptive.get('checkpoint') or {}).get('checkpoint_path')
            if ident in refs or path and checkpoint and Path(checkpoint).resolve() == path.resolve():
                raise ValueError('A running or resumable job uses this expert')
    if old is None:
        owned = bool(path and not path.is_symlink() and any(path.resolve() != (service.settings.data_dir / part).resolve() and path.resolve().is_relative_to((service.settings.data_dir / part).resolve())
                                                        for part in ('adapters', 'adaptive')))
        if owned:
            for other in store.list('adapters'):
                if other['id'] != ident and other.get('path') and Path(other['path']).resolve().is_relative_to(path.resolve()):
                    raise ValueError('Another adapter shares this artifact directory')
        target = service.settings.data_dir / 'trash' / 'adapters' / uid('artifact') if owned else None
        old = store.put('deleted_adapters', {**adapter, 'status': 'deleting', 'deleted_at': now(),
                        'original_path': str(path) if path else None, 'trash_path': str(target) if target else None,
                        'external_artifact_preserved': not owned})
    if old.get('trash_path'):
        source, target = Path(old['original_path']), Path(old['trash_path'])
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if source.exists() and target.exists():
            raise ValueError('Deletion has conflicting artifact locations; inspect before retrying')
        if source.exists():
            source.rename(target)
    with store.transaction():
        result = store.update('deleted_adapters', ident, status='deleted')
        store.delete('adapters', ident)
        if store.setting('previous_adapter') == ident:
            store.set_setting('previous_adapter', None)
        store.audit('adapter_deleted_to_trash', adapter_id=ident, trash_path=result.get('trash_path'))
    return result
