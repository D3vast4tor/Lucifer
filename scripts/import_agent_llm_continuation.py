#!/usr/bin/env python3
"""Import the verified training extension and precommit roles; never optimize."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import httpx

from lucifer.adaptive import atomic_json
from scripts.check_agent_llm_continuation import checked, sha


async def apply(root, base, base_url):
    manifest, rows, references = checked(root, base)
    signature = sha((root / 'compiled-manifest.json').read_bytes())
    staged = json.loads((root / 'validation-and-staging-receipt.json').read_text())
    if (staged['compiled_manifest_sha256'] != signature or staged['source_checks_passed'] != sum(map(len, rows.values()))
            or staged['staging']['training_groups'] != 1):
        raise ValueError('Material needs its matching successful intake staging receipt')
    previous = json.loads((base / 'import-journal.json').read_text())
    evaluation = {name: record for name, record in previous['sources'].items() if record['role'] != 'train'}
    if sorted(record['role'] for record in evaluation.values()).count('development') != 2 or len(evaluation) != 10:
        raise ValueError('Original evaluation source roles changed')
    path = root / 'import-journal.json'
    journal = json.loads(path.read_text()) if path.exists() else {'compiled_manifest_sha256': signature, 'sources': {}}
    if journal['compiled_manifest_sha256'] != signature:
        raise ValueError('Existing import belongs to other material')
    async with httpx.AsyncClient(base_url=base_url, timeout=90) as client:
        async def call(method, endpoint, **kwargs):
            response = await client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        status = await call('GET', '/system/status')
        if any(status.get(key) for key in ('agent_busy', 'chat_busy', 'learning_busy', 'meditation_busy')):
            raise ValueError('Wait for the existing chat/job without interrupting it')
        roles = {}
        # Reuse existing evaluation chats and literal targets. Do not generate,
        # attach, reread, edit or silently re-enable evaluation material here.
        for name, record in evaluation.items():
            cid = record['conversation_id']
            conversation = await call('GET', '/conversations/' + cid)
            if not conversation.get('logging_enabled') or not conversation.get('training_enabled'):
                raise ValueError('Evaluation source consent changed')
            roles[cid] = record['role']
        for item in manifest['materials']:
            name = item['id']
            title = f"Agent/LLM supervised extension — {name} [{signature[:12]}]"
            record = journal['sources'].setdefault(name, {'title': title, 'role': 'train'})
            if 'conversation_id' not in record:
                if record.get('create_pending'):
                    matches = [row for row in await call('GET', '/conversations') if row['title'] == title]
                    if len(matches) != 1:
                        raise ValueError('Conversation creation is uncertain; inspect the pending journal')
                    record['conversation_id'] = matches[0]['id']
                else:
                    record['create_pending'] = True
                    atomic_json(path, journal)
                    created = await call('POST', '/conversations', json={
                        'title': title, 'logging_enabled': True, 'training_enabled': True})
                    record['conversation_id'] = created['id']
                record.pop('create_pending', None)
                atomic_json(path, journal)
            cid = record['conversation_id']
            conversation = await call('GET', '/conversations/' + cid)
            if not conversation.get('logging_enabled') or not conversation.get('training_enabled'):
                raise ValueError('Training source consent changed; do not enable it again')
            attachments = []
            for reference, raw in references[name]:
                filename = reference.get('id', name) + '-' + reference['filename']
                document = await call('POST', f'/conversations/{cid}/knowledge', json={
                    'filename': filename, 'content_base64': base64.b64encode(raw).decode()})
                if document['sha256'] != reference['sha256']:
                    raise ValueError('Attached reference differs')
                attachments.append(document['id'])
            # Attach has per-chat content-hash deduplication. Knowledge reads
            # record interaction only with these selected workbook chunks.
            document = await call('POST', f'/conversations/{cid}/knowledge', json={
                'filename': name + '.jsonl',
                'content_base64': base64.b64encode((root / item['path']).read_bytes()).decode()})
            if document['sha256'] != item['sha256'] or document['chunks'] != len(rows[name]):
                raise ValueError('Attached training workbook differs')
            for chunk in range(0, document['chunks'], 5):
                await call('GET', f"/conversations/{cid}/knowledge/{document['id']}", params={'chunk': chunk, 'max_chunks': 5})
            record.update(reference_document_ids=attachments, document_id=document['id'], interacted=True)
            roles[cid] = 'train'
            atomic_json(path, journal)
            print(name, cid, len(rows[name]), 'checked interacted training records', flush=True)
        journal.update(imported=True, evaluation_chats_reused={name: record['conversation_id'] for name, record in evaluation.items()},
                       optimization_started=False)
        atomic_json(path, journal)
        request = {'mode': 'supervised', 'base_model': 'lucifer', 'conversation_ids': list(roles),
                   'source_roles': roles, 'dry_run': True, 'seed': 19, 'max_seq_length': 512,
                   'idempotency_key': 'supervised-extension-precommit-' + signature[:20]}
        atomic_json(root / 'precommit-request.json', request)
        result = await call('POST', '/meditation/start', json=request)
        atomic_json(root / 'precommit-start-receipt.json', result)
        print('Source-only precommit:', result['id'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--apply', action='store_true', help='Import owned public training chats and run a source-only precommit')
    args = parser.parse_args()
    root, base = args.root.resolve(strict=True), args.base.resolve(strict=True)
    if not args.apply:
        manifest, rows, _ = checked(root, base)
        print(json.dumps({'checked_records': sum(map(len, rows.values())), 'live_imported': False, 'optimization_started': False}))
        return
    asyncio.run(apply(root, base, args.base_url))


if __name__ == '__main__':
    main()
