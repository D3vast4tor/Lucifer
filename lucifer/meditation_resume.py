"""Explicit, append-only source scope extension for a stopped native run."""
from __future__ import annotations

import json
from pathlib import Path

from .meditation_sources import prepare_meditation_sources
from .schemas import MeditationResume
from .store import now


async def extended_request(service, run, key, *, conversation_ids=None, source_roles=None):
    options = MeditationResume(idempotency_key=key, conversation_ids=conversation_ids, source_roles=source_roles)
    original = run['request']
    proposed = dict(original)
    if run.get('pending_native_admission') or run.get('pending_admission'):
        raise ValueError('Resolve the pending admission before extending its source scope')
    previous_ids = original.get('conversation_ids')
    if options.conversation_ids is not None:
        if previous_ids is None:
            raise ValueError('All-eligible runs discover added chats on resume; do not narrow their scope')
        if not set(previous_ids) <= set(options.conversation_ids):
            raise ValueError('Source extension must retain all previously selected conversation IDs')
        proposed['conversation_ids'] = options.conversation_ids
    if options.source_roles is not None:
        proposed['source_roles'] = {**(original.get('source_roles') or {}), **options.source_roles}
    # This preview reads the real consent/source dependency graph and checks
    # existing reserved roles, but cannot reserve fresh groups or start workers.
    await prepare_meditation_sources(service.store, proposed, run_id=run['id'], reserve=False)
    if run.get('source_cache_path'):
        path = Path(run['source_cache_path'])
        from .adaptive import file_sha256
        if not path.is_file() or file_sha256(path) != run.get('source_bundle_sha256'):
            raise ValueError('Previous consumed source bundle changed before scope extension')
        service._check_sources(json.loads(path.read_text()))
    change = {'at': now(), 'before_conversation_ids': previous_ids,
              'after_conversation_ids': proposed.get('conversation_ids'),
              'before_source_roles': original.get('source_roles'), 'after_source_roles': proposed.get('source_roles')}
    return proposed, change
