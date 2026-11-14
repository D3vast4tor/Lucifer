"""Automatically create numerical validation targets when sources lack a split.

These internal oracle exercises are never training examples or independent
human source groups. Behavioral admission uses its separate sealed episodes.
"""
from __future__ import annotations

import copy
import json

from .behavioral import build_suite, digest


def _target(case):
    if case['family'] == 'affine_rule':
        return {'answer': case['oracle']['answer']}
    edges = case['oracle']['edges']
    best = {0: (0, [0])}
    for u, v, cost in edges:
        if u in best:
            proposal = (best[u][0] + cost, best[u][1] + [v])
            if v not in best or proposal < best[v]:
                best[v] = proposal
    if best[5][0] != case['oracle']['minimum_cost']:
        raise ValueError('Independent path target disagrees with its oracle')
    return {'path': best[5][1]}


def build_numeric_resources(seed, missing):
    if not isinstance(seed, str) or not seed or not set(missing) <= {'development', 'holdout'}:
        raise ValueError('Invalid automatic numerical resource request')
    datasets, proofs = {}, {}
    for partition in sorted(missing):
        # Separate seeds and task identities from the behavioral admission
        # suite; these small targets support drift measurement only.
        suite = build_suite(seed=digest(['numeric_validation_v1', seed, partition]), cases_per_category=2,
                            purpose='development' if partition == 'development' else 'admission')
        cases = [case for case in suite['cases'] if case['category'] == 'novel_reasoning']
        datasets[partition] = [{'messages': [{'role': 'user', 'content': case['prompt']},
                                            {'role': 'assistant', 'content': json.dumps(_target(case), sort_keys=True)}],
                                'group': 'procedural_numeric_' + partition + ':' + case['family'],
                                'fingerprint': digest(['numeric_validation_v1', partition, case])} for case in cases]
        proofs[partition] = {'suite_sha256': suite['sha256'], 'case_sha256': [digest(case) for case in cases]}
    value = {'version': 1, 'seed': seed, 'datasets': datasets, 'proofs': proofs,
             'authority': 'trusted_program_oracles', 'training_eligible': False,
             'independent_human_source_groups_claimed': False}
    value['sha256'] = digest(value)
    return value


def validate_numeric_resources(value):
    if value != build_numeric_resources(value['seed'], value['datasets'].keys()):
        raise ValueError('Automatic numerical targets or independent oracle proof changed')
    return True


def attach_numeric_validation(bundle, seed):
    missing = [partition for partition, key in [('development', 'development_examples'), ('holdout', 'holdout_examples')]
               if not bundle[key]]
    if not missing:
        return bundle  # Preserve existing native feature/data identities.
    value = build_numeric_resources(seed, missing)
    result = copy.deepcopy(bundle)
    result['automatic_validation'] = value
    result['manifest']['automatic_validation_sha256'] = value['sha256']
    result['manifest']['manifest_sha256'] = digest({k: v for k, v in result['manifest'].items() if k != 'manifest_sha256'})
    result['metrics']['automatic_numerical_validation'] = {k: len(v) for k, v in value['datasets'].items()}
    return result
