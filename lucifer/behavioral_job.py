"""Shared job resources and completion-aware model runner for both training modes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .adaptive import atomic_json
from .behavioral import digest, freeze_suite, measure_suite, validate_suite
from .store import now
from .training_contract import master_prompt, prompt_receipt


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_resources(directory, *, mode, training_prompts=(), cases_per_category=2, purpose='admission'):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError('Behavioral resources must be an owned directory')
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    suite_path, prompt_path = directory / 'suite.json', directory / 'training-master.txt'
    prompt = master_prompt(mode)
    if prompt_path.exists() and prompt_path.read_text() != prompt:
        raise ValueError('Pinned training master prompt changed')
    if not prompt_path.exists():
        prompt_path.write_text(prompt)
        prompt_path.chmod(0o600)
    suite = freeze_suite(suite_path, cases_per_category=cases_per_category,
                         purpose=purpose, training_prompts=training_prompts)
    binding = {'protocol': suite['protocol'], 'path': str(suite_path.resolve()),
               'file_sha256': file_hash(suite_path), 'suite_sha256': suite['sha256'],
               'mode': mode, 'master_prompt': prompt_receipt(mode), 'master_prompt_path': str(prompt_path.resolve()),
               'case_count': len(suite['cases']), 'categories': sorted({c['category'] for c in suite['cases']}),
               'budget': suite['budget'], 'purpose': purpose}
    binding['verifier_files'] = {name: file_hash(Path(__file__).with_name(name))
        for name in ('behavioral.py', 'behavioral_code.py', 'behavioral_job.py')}
    path = directory / 'resources.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if {k: v for k, v in previous.items() if k != 'frozen_at'} != binding:
            raise ValueError('Behavioral job resources changed')
        return previous
    binding['frozen_at'] = now()
    atomic_json(path, binding)
    return binding


def load_resources(binding):
    if binding.get('verifier_files') != {name: file_hash(Path(__file__).with_name(name))
            for name in ('behavioral.py', 'behavioral_code.py', 'behavioral_job.py')}:
        raise ValueError('Pinned behavioral verifier implementation changed')
    if file_hash(binding['path']) != binding['file_sha256']:
        raise ValueError('Sealed behavioral suite file changed')
    if file_hash(binding['master_prompt_path']) != binding['master_prompt']['sha256']:
        raise ValueError('Sealed training master prompt file changed')
    suite = json.loads(Path(binding['path']).read_text())
    validate_suite(suite)
    if suite['sha256'] != binding['suite_sha256'] or suite['budget'] != binding['budget']:
        raise ValueError('Behavioral suite differs from its job binding')
    return suite


async def measure_selection(learning, binding, directory, *, model_id, adapter=None, artifact_sha256=None, progress=None):
    """Caller owns the exclusive model lease and restoration for this selection."""
    suite = load_resources(binding)
    current = learning.manager.current()
    if not current or current['id'] != model_id or current.get('backend') == 'demo':
        raise ValueError('A real selected model is required for behavioral evaluation')
    registry = getattr(learning.manager, 'capabilities', None)
    snapshot = registry.get(model_id) if registry is not None else None
    selection = {'model_id': model_id, 'profile_revision': snapshot.profile_revision if snapshot is not None else None,
                 'adapter_id': adapter.get('id') if adapter else None,
                 'adapter_artifact_sha256': adapter.get('checkpoint_manifest_sha256') if adapter else None,
                 'selected_artifact_sha256': artifact_sha256,
                 'suite_sha256': suite['sha256'], 'budget': suite['budget']}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / 'selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('Cached episode outcomes belong to another model selection')
    if not path.exists():
        atomic_json(path, selection)

    async def reply(messages, budget):
        count = await learning._evaluation_token_count(messages, model_id, budget)
        return await learning._evaluation_answer(messages, adapter, model_id, budget, count)

    report = await measure_suite(suite, root, reply, progress=progress)
    report['selection'] = selection
    atomic_json(root / 'measurement.json', report)
    return report


def resource_identity(binding):
    """Compact identity suitable for parent/candidate artifact admission binding."""
    load_resources(binding)
    return {'protocol': binding['protocol'], 'suite_sha256': binding['suite_sha256'],
            'file_sha256': binding['file_sha256'], 'master_prompt': binding['master_prompt'],
            'resource_binding_sha256': digest(binding)}


def evidence_files(*directories):
    """Hash the completed selection, case journals and projected measurements."""
    result = []
    for value in directories:
        root = Path(value)
        for path in sorted(root.rglob('*.json')):
            if path.is_symlink():
                raise ValueError('Behavioral evidence must remain in owned regular files')
            result.append({'path': str(path.resolve()), 'sha256': file_hash(path)})
    return result


def verify_evidence(entries):
    if not entries or any(Path(item['path']).is_symlink() or file_hash(item['path']) != item['sha256'] for item in entries):
        raise ValueError('Behavioral episode evidence changed or disappeared')
