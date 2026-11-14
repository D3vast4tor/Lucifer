#!/usr/bin/env python3
"""Check sealed continuation data and stage real knowledge intake, without training."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from lucifer.adaptive import atomic_json
from lucifer.chat import ChatService
from lucifer.learning import training_python
from lucifer.meditation_sources import prepare_meditation_sources, revalidate_meditation_sources
from lucifer.meditation_worker import load_primary_tokenizer
from lucifer.problem_solution import parse_record, source_family
from lucifer.store import Store, now


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(root, base):
    raw = (root / 'compiled-manifest.json').read_bytes()
    manifest = json.loads(raw)
    old_raw = (base / 'compiled-manifest.json').read_bytes()
    if sha(old_raw) != manifest['original_manifest_sha256']:
        raise ValueError('Original manifest changed')
    old_train = {item['id']: item for item in json.loads(old_raw)['materials'] if item['role'] == 'train'}
    if sha((root / 'selection.json').read_bytes()) != manifest['selection_sha256']:
        raise ValueError('Selection changed')
    if sha((root / 'executed-builder.py').read_bytes()) != manifest['builder_sha256']:
        raise ValueError('Executed builder changed')
    references, by_family = {}, {name: [] for name in old_train}
    for name in old_train:
        ref = json.loads((base / 'references' / name / 'receipt.json').read_text())
        source = (base / 'references' / name / ref['filename']).read_bytes()
        if sha(source) != ref['sha256']:
            raise ValueError('Original source changed')
        references[sha(source)] = (ref, source)
        by_family[name].append((ref, source))
    for item in manifest['additional_sources']:
        folder = root / 'references' / item['id']
        if sha((folder / 'source-receipt.json').read_bytes()) != item['source_receipt_sha256']:
            raise ValueError('New source receipt changed')
        ref = json.loads((folder / 'source-receipt.json').read_text())
        source = (folder / ref['filename']).read_bytes()
        if sha(source) != ref['sha256'] or ref != item['source']:
            raise ValueError('New source changed')
        if item['id'] not in old_train or source_family(ref['url']) != old_train[item['id']]['source_family']:
            raise ValueError('Nontraining source family')
        references[sha(source)] = (ref, source)
        by_family[item['id']].append((ref, source))
    rows = {}
    for item in manifest['materials']:
        if item['role'] != 'train' or item['id'] not in old_train:
            raise ValueError('Nontraining workbook')
        raw = (root / item['path']).read_bytes()
        if sha(raw) != item['sha256']:
            raise ValueError('Training workbook changed')
        values = [parse_record(line) for line in raw.decode().splitlines()]
        if len(values) != item['records']:
            raise ValueError('Record count changed')
        for row in values:
            proof_raw = (root / 'verification' / (row['id'] + '.json')).read_bytes()
            if sha(proof_raw) != row['verification']['receipt_sha256']:
                raise ValueError('Verification receipt changed')
            proof = json.loads(proof_raw)
            if (proof['answer_sha256'] != sha(row['answer'].encode())
                    or proof['question_sha256'] != sha(row['question'].encode())):
                raise ValueError('Question or checked answer changed')
            for source in row['sources']:
                ref, _ = references[source['sha256']]
                if ref['url'] != source['url'] or source_family(source['url']) != old_train[item['id']]['source_family']:
                    raise ValueError('Workbook source binding changed')
            method = row['verification']['method']
            if method == 'exact_source_continuation':
                text = references[proof['source_sha256']][1].decode()
                start, split, end = (proof[key] for key in ('start', 'split', 'end'))
                if (not 0 <= start < split < end <= len(text) or row['answer'] != text[split:end]
                        or sha(text[start:split].encode()) != proof['prefix_sha256']
                        or not row['question'].endswith(text[start:split]) or proof['provider_reasoning'] is not False):
                    raise ValueError('Literal continuation does not match its source')
            elif method != 'trusted_local_fixture':
                raise ValueError('Unknown verification method')
            if manifest['shared_dataset_lineage'] not in row['lineage']:
                raise ValueError('Authored material lost shared lineage')
        rows[item['id']] = values
    return manifest, rows, by_family


async def stage(root, manifest, rows, by_family):
    with tempfile.TemporaryDirectory(prefix='lucifer-curriculum-v2-stage-') as temp:
        store = Store(Path(temp) / 'stage.sqlite')
        chat, roles = ChatService(store, SimpleNamespace()), {}
        try:
            for item in manifest['materials']:
                ident = item['id']
                cid = chat.create(title='Staged training-only ' + ident, training_enabled=True)['id']
                roles[cid] = 'train'
                for ref, raw in by_family[ident]:
                    await chat.knowledge.attach(cid, filename=ref['filename'], content_base64=base64.b64encode(raw).decode())
                doc = await chat.knowledge.attach(cid, filename=ident + '.jsonl',
                    content_base64=base64.b64encode((root / item['path']).read_bytes()).decode())
                if doc['chunks'] != len(rows[ident]):
                    raise ValueError('Knowledge intake changed record boundaries')
                for chunk in range(0, doc['chunks'], 5):
                    chat.knowledge.read(cid, doc['id'], chunk, 5)
            bundle = await prepare_meditation_sources(store, {'mode': 'supervised', 'source_roles': roles})
            if (not revalidate_meditation_sources(store, bundle)
                    or len(bundle['examples']) != sum(len(items) for items in rows.values())
                    or len(bundle['manifest']['groups']) != 1
                    or any(group['partition'] != 'train' for group in bundle['manifest']['groups'])):
                raise ValueError('Staged sources lost consent, examples or shared lineage')
            return {'staged': True, 'examples': len(bundle['examples']), 'training_groups': 1,
                    'documents': len(store.list('knowledge_documents')), 'evaluation_chats_modified': False}
        finally:
            store.close()


def token_counts(rows, runtime_profile):
    profile = json.loads(runtime_profile.read_text())
    tokenizer = load_primary_tokenizer({'tokenizer_path': profile['tokenizer'], 'runtime_profile': str(runtime_profile)})
    counts, by_family = {}, {}
    for name, items in rows.items():
        for row in items:
            messages = [{'role': 'user', 'content': row['question']}, {'role': 'assistant', 'content': row['answer']}]
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=False)
            if full[:len(prompt)] != prompt or len(full) <= len(prompt):
                raise ValueError('Native assistant target prefix differs')
            targets = len(full) - len(prompt)
            method = row['verification']['method']
            counts[method] = counts.get(method, 0) + targets
            by_family[name] = by_family.get(name, 0) + targets
    return {'native_assistant_target_tokens': sum(counts.values()), 'target_tokens_by_method': counts,
            'target_tokens_by_training_family': by_family, 'tokenizer_provenance': tokenizer.provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--runtime-profile', type=Path, required=True)
    parser.add_argument('--count-only', action='store_true', help='Use the native training environment for tokenizer dependencies')
    args = parser.parse_args()
    root, base = args.root.resolve(strict=True), args.base.resolve(strict=True)
    manifest, rows, refs = checked(root, base)
    if args.count_only:
        print(json.dumps(token_counts(rows, args.runtime_profile)))
        return
    staged = asyncio.run(stage(root, manifest, rows, refs))
    # Core document intake and the native tokenizer deliberately use separate
    # environments, just as in Meditation. No model or GPU is initialized.
    process = subprocess.run([training_python(), str(Path(__file__).resolve()), '--count-only',
        '--root', str(root), '--base', str(base), '--runtime-profile', str(args.runtime_profile.resolve())],
        capture_output=True, text=True, check=True)
    counts = json.loads(process.stdout)
    report = {'at': now(), 'compiled_manifest_sha256': sha((root / 'compiled-manifest.json').read_bytes()),
              'source_checks_passed': sum(len(items) for items in rows.values()), 'staging': staged,
              **counts,
              'heldout_targets_read': False, 'live_imported': False, 'optimization_started': False,
              'quality_improvement_proven': False}
    atomic_json(root / 'validation-and-staging-receipt.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'tokenizer_provenance'}, indent=2))


if __name__ == '__main__':
    main()
