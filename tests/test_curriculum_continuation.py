"""Checked source continuations cannot escape provenance or admit forged targets."""
import json

import pytest

from scripts.check_agent_llm_continuation import checked, sha
from scripts.prepare_agent_llm_continuation import dump, paragraphs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (dump(value) + '\n').encode()
    path.write_bytes(raw)
    return sha(raw)


@pytest.fixture
def pack(tmp_path):
    base, root = tmp_path / 'base', tmp_path / 'new'
    text = 'A technical model prefix. The checked continuation describes cache memory and valid tool schemas.'
    prefix = 'A technical model prefix.'
    source = {'id': 'technical', 'url': 'https://example.org/technical', 'filename': 'source.txt',
              'sha256': sha(text.encode())}
    write(base / 'references/technical/receipt.json', source)
    (base / 'references/technical/source.txt').write_text(text)
    old_sha = write(base / 'compiled-manifest.json', {'materials': [
        {'id': 'technical', 'role': 'train', 'source_family': 'origin:example.org'}]})
    root.mkdir()
    builder = root / 'executed-builder.py'
    builder.write_text('This file is checked as bytes, never executed.')
    selection_sha = write(root / 'selection.json', {'references': ['technical']})
    lineage = 'a' * 64
    question = 'Complete this quoted passage: ' + prefix
    answer = text[len(prefix):]
    proof = {'source_sha256': source['sha256'], 'start': 0, 'split': len(prefix), 'end': len(text),
             'prefix_sha256': sha(prefix.encode()), 'answer_sha256': sha(answer.encode()),
             'question_sha256': sha(question.encode()), 'provider_reasoning': False}
    proof_sha = write(root / 'verification/technical-0.json', proof)
    row = {'format': 'lucifer.problem-solution.v1', 'id': 'technical-0', 'question': question, 'answer': answer,
           'sources': [{'url': source['url'], 'sha256': source['sha256']}], 'lineage': [lineage],
           'verification': {'method': 'exact_source_continuation', 'receipt_sha256': proof_sha}}
    workbook_sha = write(root / 'materials/technical.jsonl', row)
    manifest = {'original_manifest_sha256': old_sha, 'builder_sha256': sha(builder.read_bytes()),
                'selection_sha256': selection_sha, 'shared_dataset_lineage': lineage, 'additional_sources': [],
                'materials': [{'id': 'technical', 'role': 'train', 'path': 'materials/technical.jsonl',
                               'sha256': workbook_sha, 'records': 1}]}
    write(root / 'compiled-manifest.json', manifest)
    return base, root, row, proof, manifest


def test_proof_checks_source_offsets_even_if_all_answer_and_manifest_hashes_are_resealed(pack):
    base, root, row, proof, manifest = pack
    assert checked(root, base)[1]['technical'][0]['answer'] == row['answer']
    row['answer'] = 'Forged answer with internally consistent digests.'
    proof['answer_sha256'] = sha(row['answer'].encode())
    row['verification']['receipt_sha256'] = write(root / 'verification/technical-0.json', proof)
    manifest['materials'][0]['sha256'] = write(root / 'materials/technical.jsonl', row)
    write(root / 'compiled-manifest.json', manifest)
    with pytest.raises(ValueError, match='does not match its source'):
        checked(root, base)


def test_source_tampering_and_dropped_dataset_lineage_are_rejected(pack):
    base, root, row, proof, manifest = pack
    row['lineage'] = ['b' * 64]
    manifest['materials'][0]['sha256'] = write(root / 'materials/technical.jsonl', row)
    write(root / 'compiled-manifest.json', manifest)
    with pytest.raises(ValueError, match='shared lineage'):
        checked(root, base)
    (base / 'references/technical/source.txt').write_text('Changed source')
    with pytest.raises(ValueError, match='Original source changed'):
        checked(root, base)


def test_nontraining_workbook_rejected_before_reading_its_contents(pack):
    base, root, row, proof, manifest = pack
    manifest['materials'][0].update(role='heldout', path='must-not-be-opened')
    write(root / 'compiled-manifest.json', manifest)
    with pytest.raises(ValueError, match='Nontraining workbook'):
        checked(root, base)


def test_prose_selection_keeps_original_offsets_and_excludes_fenced_instructions():
    selection = {'paragraph_characters': [140, 1200], 'paragraph_words': [35, 180], 'prefix_words': 16}
    prose = ('The model stores data in a cache so that each new token can reuse previous computations. '
             'The cache memory grows with stored tokens while the selected implementation controls its layout. '
             'A client validates the supported cache type before configuring the model.')
    hidden = '```text\n' + prose + '\n```'
    text = '<!--\n' + prose + '\n-->\n\n' + hidden + '\n\n' + prose
    spans = list(paragraphs(text, selection))
    assert len(spans) == 1
    start, end, split = spans[0]
    assert text[start:end] == prose and start > text.index('```') and start < split < end
    assert json.loads(json.dumps(text[split:end])) == text[split:end]


@pytest.mark.asyncio
@pytest.mark.parametrize('consent', [True, False])
async def test_import_recovers_pending_creation_and_never_rewrites_evaluation_material(pack, monkeypatch, consent):
    import base64
    import httpx
    from scripts import import_agent_llm_continuation as importer
    base, root, row, proof, manifest = pack
    signature = sha((root / 'compiled-manifest.json').read_bytes())
    write(root / 'validation-and-staging-receipt.json', {'compiled_manifest_sha256': signature,
        'source_checks_passed': 1, 'staging': {'training_groups': 1}})
    evaluation = {f'eval-{i}': {'role': 'development' if i < 2 else 'heldout', 'conversation_id': f'eval-{i}'}
                  for i in range(10)}
    write(base / 'import-journal.json', {'sources': evaluation})
    title = f'Agent/LLM supervised extension — technical [{signature[:12]}]'
    write(root / 'import-journal.json', {'compiled_manifest_sha256': signature, 'sources': {
        'technical': {'role': 'train', 'title': title, 'create_pending': True}}})
    requests = []
    def handle(request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, path, body))
        if path == '/system/status':
            return httpx.Response(200, json={})
        if path == '/conversations':
            assert request.method == 'GET'  # Recovery must not create another chat.
            return httpx.Response(200, json=[{'id': 'training', 'title': title}])
        if path.startswith('/conversations/eval-'):
            assert request.method == 'GET' and '/knowledge' not in path
            return httpx.Response(200, json={'logging_enabled': True, 'training_enabled': consent})
        if path == '/conversations/training':
            return httpx.Response(200, json={'logging_enabled': True, 'training_enabled': True})
        if path == '/conversations/training/knowledge':
            content = base64.b64decode(body['content_base64'])
            return httpx.Response(200, json={'id': 'workbook' if body['filename'].endswith('.jsonl') else 'reference',
                                            'sha256': sha(content), 'chunks': 1})
        if path.endswith('/knowledge/workbook'):
            return httpx.Response(200, json={'chunks': []})
        if path == '/meditation/start':
            assert body['dry_run'] is True
            return httpx.Response(202, json={'id': 'source-only-precommit'})
        raise AssertionError((request.method, path))
    original_client = httpx.AsyncClient
    monkeypatch.setattr(importer.httpx, 'AsyncClient', lambda **kwargs: original_client(
        transport=httpx.MockTransport(handle), **kwargs))
    if not consent:
        with pytest.raises(ValueError, match='consent changed'):
            await importer.apply(root, base, 'http://test')
        assert not any(method == 'POST' for method, _, _ in requests)
        return
    await importer.apply(root, base, 'http://test')
    first = json.loads((root / 'precommit-request.json').read_text())
    assert first['source_roles'] == {**{k: v['role'] for k, v in evaluation.items()}, 'training': 'train'}
    assert first['dry_run'] is True
    await importer.apply(root, base, 'http://test')
    assert json.loads((root / 'precommit-request.json').read_text()) == first
    assert not any(method == 'PATCH' or path.endswith('/resume') for method, path, _ in requests)
