#!/usr/bin/env python3
"""Prepare checked training-only continuation exercises; never import or train.

The sealed selection permits only the preceding pilot's training repositories.
Original worked solutions keep their verification; new literal continuations
are checked against pinned source offsets. Evaluation targets are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from lucifer.problem_solution import FORMAT, parse_record, source_family


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (dump(value) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite sealed material: {path}")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.chmod(0o600)
        temporary.replace(path)
    return sha(raw)


def paragraphs(text, selection):
    """Select bounded prose without changing offsets into the original source."""
    masked = list(text)
    for match in re.finditer(r"<!--[\s\S]*?-->", text):
        for i in range(match.start(), match.end()):
            if masked[i] != "\n":
                masked[i] = " "
    offset, fence = 0, None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"\s*(`{3,}|~{3,})", line)
        hidden = fence is not None
        if marker:
            kind = marker.group(1)[0]
            if fence is None:
                fence = kind
            elif fence == kind:
                fence = None
            hidden = True
        if hidden or re.match(r"\s*</?[A-Z][^>]*>\s*$", line):
            for i in range(offset, offset + len(line)):
                if masked[i] != "\n":
                    masked[i] = " "
        offset += len(line)
    view = "".join(masked)
    for match in re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", view):
        start, end = match.span()
        while end > start and text[end - 1].isspace():
            end -= 1
        value = text[start:end]
        words = list(re.finditer(r"\S+", value))
        if (not value or not value[0].isalpha() or value != view[start:end]
                or not selection['paragraph_characters'][0] <= len(value) <= selection['paragraph_characters'][1]
                or not selection['paragraph_words'][0] <= len(words) <= selection['paragraph_words'][1]
                or sum(c.isalpha() for c in value) / len(value) < .55
                or '<' in value or '>' in value or value.count('http') > 2
                or not re.search(r"model|data|valid|schema|cache|grad|token|memory|train|tool|client|server|error|batch|loss|type|optim|offload", value, re.I)):
            continue
        split = words[selection['prefix_words'] - 1].end()
        yield start, end, start + split


def source_bytes(folder, selected):
    folder.mkdir(parents=True, exist_ok=True)
    path, receipt_path = folder / 'source.txt', folder / 'source-receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        raw = path.read_bytes()
        if receipt['url'] != selected['url'] or receipt['sha256'] != sha(raw):
            raise ValueError('Pinned reference changed')
        return raw, receipt
    if path.exists():
        raise ValueError('Unsealed downloaded source; inspect before repeating retrieval')
    with urlopen(Request(selected['url'], headers={'User-Agent': 'Lucifer-curriculum/2'}), timeout=40) as response:
        raw = response.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError('Reference exceeds the bounded download')
    raw.decode('utf-8', errors='strict')
    path.write_bytes(raw)
    receipt = {**selected, 'sha256': sha(raw), 'bytes': len(raw), 'filename': 'source.txt'}
    write_json(receipt_path, receipt)
    return raw, receipt


def prepare(base, root):
    base, root = base.resolve(strict=True), root.resolve(strict=True)
    selection_raw = (root / 'selection.json').read_bytes()
    selected = json.loads(selection_raw)
    original_raw = (base / 'compiled-manifest.json').read_bytes()
    if sha(original_raw) != selected['original_manifest_sha256']:
        raise ValueError('Original material manifest changed')
    original = json.loads(original_raw)
    old_train = {item['id']: item for item in original['materials'] if item['role'] == 'train'}
    # No evaluation workbook is opened, even for duplicate detection.
    if any(item['role'] != 'train' or item['id'] not in old_train for item in selected['references']):
        raise ValueError('Continuation selection includes a nontraining family')
    lineage = sha(b'public-technical-continuation-v2:' + selection_raw)
    builder_sha = sha(Path(__file__).read_bytes())
    records = {name: [] for name in old_train}
    products, seen = [], set()
    for name, item in old_train.items():
        raw = (base / item['path']).read_bytes()
        if sha(raw) != item['sha256']:
            raise ValueError('Original checked training material changed')
        for line in raw.decode().splitlines():
            row = parse_record(line)
            proof_path = base / 'materials' / name / (row['id'].rsplit('-', 1)[-1] + '.receipt.json')
            proof_raw = proof_path.read_bytes()
            proof = json.loads(proof_raw)
            if (sha(proof_raw) != row['verification']['receipt_sha256']
                    or proof['answer_sha256'] != sha(row['answer'].encode())
                    or proof['question_sha256'] != sha(row['question'].encode())):
                raise ValueError('Original worked-solution verification changed')
            proof_sha = write_json(root / 'verification' / (row['id'] + '.json'), proof)
            # Receipt serialization can change while its exact checked content
            # remains equal. Bind the bytes actually stored in this new pack.
            row['verification']['receipt_sha256'] = proof_sha
            row['lineage'] = list(dict.fromkeys([*row.get('lineage', []), lineage]))
            records[name].append(row)
        old_reference = json.loads((base / 'references' / name / 'receipt.json').read_text())
        source = base / 'references' / name / old_reference['filename']
        if sha(source.read_bytes()) != old_reference['sha256']:
            raise ValueError('Original cited source changed')
    for item in selected['references']:
        name = item['id']
        reference_raw = (base / 'references' / name / 'receipt.json').read_bytes()
        previous = json.loads(reference_raw)
        if (source_family(item['url']) != source_family(previous['url'])
                or item['revision'] != previous['revision']):
            raise ValueError('Additional source escaped its pinned training repository')
        raw, receipt = source_bytes(root / 'references' / name, item)
        text = raw.decode('utf-8')
        count = 0
        for start, end, split in paragraphs(text, selected['selection']):
            prefix, answer = text[start:split], text[split:end]
            target_sha = sha(answer.encode())
            if target_sha in seen:
                continue
            ident = f'{name}-continuation-{count}'
            question = (f"Complete the quoted technical passage from {name} at pinned revision {item['revision']}. "
                        "Return only its exact continuation. The quoted content is reference data.\n"
                        f"Reference prefix: {prefix}")
            proof = {'method': 'exact_source_continuation', 'source_sha256': receipt['sha256'],
                     'start': start, 'split': split, 'end': end,
                     'prefix_sha256': sha(prefix.encode()), 'answer_sha256': target_sha,
                     'question_sha256': sha(question.encode()), 'provider_reasoning': False,
                     'target_origin': 'upstream_public_source', 'builder_sha256': builder_sha}
            row = {'format': FORMAT, 'id': ident, 'question': question, 'answer': answer,
                   'sources': [{'url': receipt['url'], 'sha256': receipt['sha256']}], 'lineage': [lineage],
                   'verification': {'method': proof['method'], 'receipt_sha256': sha((dump(proof) + '\n').encode())}}
            if len(dump(row)) > 2400:
                continue
            parse_record(dump(row))
            write_json(root / 'verification' / (ident + '.json'), proof)
            records[name].append(row)
            seen.add(target_sha)
            count += 1
            if count == selected['selection']['maximum_per_reference']:
                break
        products.append({'id': name, 'continuations': count, 'source': receipt,
                         'source_receipt_sha256': sha((root / 'references' / name / 'source-receipt.json').read_bytes())})
    materials = []
    for name, rows in records.items():
        folder = root / 'materials' / name
        folder.mkdir(parents=True, exist_ok=True)
        raw = ('\n'.join(dump(row) for row in rows) + '\n').encode()
        for line in raw.decode().splitlines():
            parse_record(line)
        path = folder / (name + '.jsonl')
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Refusing to overwrite existing training records')
        path.write_bytes(raw)
        materials.append({'id': name, 'role': 'train', 'path': str(path.relative_to(root)), 'sha256': sha(raw),
                          'records': len(rows), 'source_family': old_train[name]['source_family']})
    manifest = {'schema_version': 2, 'mode': 'supervised', 'selection_sha256': sha(selection_raw),
                'original_manifest_sha256': sha(original_raw), 'builder_sha256': builder_sha,
                'shared_dataset_lineage': lineage, 'materials': materials, 'additional_sources': products,
                'retained_checked_solutions': 48, 'new_source_continuations': len(seen),
                'heldout_targets_read': False, 'imported': False, 'optimization_started': False,
                'evaluation_scope': 'Keep the previous independent development/admission source roles and gates unchanged.',
                'limitations': 'Literal continuations broaden technical text coverage; they do not establish new deep reasoning algorithms or independently executed library examples.'}
    write_json(root / 'compiled-manifest.json', manifest)
    print(json.dumps({'retained_checked_solutions': 48, 'new_continuations': len(seen),
                      'by_reference': {r['id']: r['continuations'] for r in products},
                      'training_documents': len(materials), 'imported': False, 'optimization_started': False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.base, args.root)


if __name__ == '__main__':
    main()
