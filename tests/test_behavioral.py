import copy
import json

import pytest

from lucifer.behavioral import (BUDGET, CATEGORIES, Episode, build_suite, check_code, compare,
                                digest, freeze_suite, measure_suite, run_case, validate_suite)
from lucifer.training_contract import master_prompt, prompt_receipt


def cases():
    return build_suite(seed='test-independent-outcomes', cases_per_category=2)['cases']


def test_oracles_recompute_and_suite_cannot_be_forged_or_replaced(tmp_path):
    suite = freeze_suite(tmp_path / 'suite.json', seed='precommitted')
    assert set(c['category'] for c in suite['cases']) == set(CATEGORIES)
    assert validate_suite(suite)
    assert freeze_suite(tmp_path / 'suite.json') == suite
    with pytest.raises(ValueError, match='cannot be replaced'):
        freeze_suite(tmp_path / 'suite.json', seed='easier-after-training')
    altered = copy.deepcopy(suite)
    altered['cases'][0]['oracle']['answer'] += 1
    altered['sha256'] = digest({k: v for k, v in altered.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='sealed generator'):
        validate_suite(altered)
    with pytest.raises(ValueError, match='overlaps'):
        build_suite(seed='precommitted', training_prompts=[suite['cases'][0]['prompt']])


@pytest.mark.parametrize('source', [
    'def solve(xs):\n    out=[]\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n',
    'def solve(xs):\n    return list(dict.fromkeys(xs))\n',
    'def solve(xs):\n    seen=set()\n    result=[]\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result\n',
])
def test_hidden_code_tests_accept_algorithms_and_reject_original_bug(source):
    case = next(c for c in cases() if c['family'] == 'stable_unique')
    assert check_code(source, case['oracle']['tests'])['passed']
    assert not check_code(case['state']['files']['solution.py'], case['oracle']['tests'])['passed']
    assert len(case['oracle']['tests']) > len(case['oracle']['visible_tests'])


@pytest.mark.parametrize('source', [
    'import os\ndef solve(xs):\n    return os.listdir("/")',
    'def solve(xs):\n    return solve.__globals__',
    'def solve(xs):\n    return open("/etc/passwd").read()',
    'def solve(xs):\n    while True: pass',
])
def test_code_cannot_access_host_hidden_context_or_run_forever(source):
    assert not check_code(source, [{'args': [[]], 'expected': []}])['passed']


def test_changed_schema_state_reads_and_replays_are_transactional(tmp_path):
    case = next(c for c in cases() if c['family'] == 'changed_tool_schema')
    episode = Episode(case, tmp_path / 'episode')
    oracle = case['oracle']
    before = episode.call('inspect_state', {})
    assert episode.call('reserve', {'sku': oracle['sku'], 'quantity': oracle['quantity'], 'key': 'order'})['error'] == 'schema_changed_before_effect'
    args = {'sku': oracle['sku'], 'units': oracle['quantity'], 'key': 'order'}
    result = episode.call('reserve', args)
    assert episode.call('reserve', args) == result
    after = episode.call('inspect_state', {})
    assert before['stock'] != after['stock'] and after['mutation_count'] == 1
    # Same key may be used at distinct endpoints; retries cannot double effects.
    episode.call('ship', {'key': 'order'})
    episode.call('ship', {'key': 'order'})
    assert episode.score()['passed'] and episode.record['state']['mutation_count'] == 2
    restored = Episode(case, tmp_path / 'episode')
    assert restored.call('reserve', args) == result
    assert restored.score()['passed']


def test_transient_receipt_failure_does_not_cache_a_permanent_failure(tmp_path):
    case = next(c for c in cases() if c['family'] == 'interrupted_payment')
    episode = Episode(case, tmp_path / 'episode')
    key = case['oracle']['key']
    assert episode.call('get_receipt', {'key': key})['error'] == 'transient_lookup_failure_before_effect'
    assert episode.call('get_receipt', {'key': key})['receipt']['status'] == 'completed'
    episode.call('transfer', {'amount': case['oracle']['amount'], 'key': key})
    episode.call('acknowledge', {'key': key})
    assert episode.score()['passed'] and episode.record['state']['mutation_count'] == 1


@pytest.mark.asyncio
async def test_completed_episode_is_reused_and_missing_transport_evidence_never_passes(tmp_path):
    case = cases()[0]
    calls = []
    async def answer(messages, budget):
        calls.append(messages)
        assert 'oracle' not in messages[0]['content']
        return {'complete': True, 'answer': json.dumps(case['oracle'])}
    result = await run_case(case, tmp_path / 'case', answer, BUDGET)
    assert result['passed'] and result['provider_reasoning_retained'] is False
    assert await run_case(case, tmp_path / 'case', answer, BUDGET) == result and len(calls) == 1
    result_path = tmp_path / 'case/result.json'
    result_path.unlink()
    assert await run_case(case, tmp_path / 'case', answer, BUDGET) == result and len(calls) == 1
    result_path.write_text(json.dumps({**result, 'passed': False}))
    with pytest.raises(ValueError, match='outcome receipt changed'):
        await run_case(case, tmp_path / 'case', answer, BUDGET)
    async def incomplete(messages, budget):
        return {'complete': False, 'answer': 'not a completed provider answer'}
    failed = await run_case(case, tmp_path / 'incomplete', incomplete, BUDGET)
    assert not failed['complete'] and not failed['passed']


@pytest.mark.asyncio
async def test_outcome_comparison_requires_equal_cases_budgets_and_real_gain(tmp_path):
    suite = build_suite(seed='all-categories')
    async def wrong(messages, budget):
        return {'complete': True, 'answer': '{"tool":"finish","arguments":{}}'}
    baseline = await measure_suite(suite, tmp_path / 'base', wrong)
    candidate = copy.deepcopy(baseline)
    report = compare(suite, baseline, candidate)
    assert report['complete'] and not report['passed'] and report['quality_status'] == 'not_improved'
    candidate['cases'].pop()
    assert not compare(suite, baseline, candidate)['complete']
    candidate = copy.deepcopy(baseline)
    candidate['budget'] = {**BUDGET, 'max_calls': 16}
    with pytest.raises(ValueError, match='same sealed tasks and budgets'):
        compare(suite, baseline, candidate)


def test_adequate_paired_gain_can_pass_but_regressions_or_forged_rows_cannot():
    suite = build_suite(seed='paired-outcome-policy')
    rows = [{'id': c['id'], 'category': c['category'], 'family': c['family'],
             'case_sha256': digest(c), 'budget': suite['budget'], 'complete': True, 'passed': False}
            for c in suite['cases']]
    base = {'suite_sha256': suite['sha256'], 'budget': suite['budget'], 'cases': rows}
    candidate = copy.deepcopy(base)
    for row in candidate['cases'][:6]:
        row['passed'] = True
    report = compare(suite, base, candidate)
    assert report['passed'] and report['paired_test']['p_value'] == 1 / 64
    base['cases'][-1]['passed'] = True
    assert not compare(suite, base, candidate)['passed']
    base['cases'][-1]['passed'] = False
    candidate['cases'][0]['case_sha256'] = 'forged'
    assert not compare(suite, base, candidate)['complete']


def test_shared_master_prompt_keeps_modes_and_hidden_evidence_separate():
    sleep, meditation = master_prompt('sleep'), master_prompt('meditation')
    assert sleep.startswith('LUCIFER_TRAINING_MASTER_V1')
    assert 'adapters/experts' in sleep and 'PRIMARY model output head' in meditation
    assert 'independent validation' in sleep and 'private reasoning' in meditation
    assert prompt_receipt('sleep')['sha256'] != prompt_receipt('meditation')['sha256']
    with pytest.raises(ValueError):
        master_prompt('self_approve')


def test_job_resources_bind_master_prompt_and_private_suite_before_model_calls(tmp_path):
    from lucifer.behavioral_job import load_resources, prepare_resources, resource_identity
    binding = prepare_resources(tmp_path / 'resources', mode='sleep', cases_per_category=1)
    suite = load_resources(binding)
    assert len(suite['cases']) == 4
    assert resource_identity(binding)['suite_sha256'] == suite['sha256']
    assert prepare_resources(tmp_path / 'resources', mode='sleep', cases_per_category=1) == binding
    with pytest.raises(ValueError, match='master prompt changed'):
        prepare_resources(tmp_path / 'resources', mode='meditation', cases_per_category=1)
    from pathlib import Path
    Path(binding['path']).write_text('{}')
    with pytest.raises(ValueError, match='suite file changed'):
        load_resources(binding)


@pytest.mark.asyncio
async def test_automatic_numeric_resources_need_no_authored_holdout_chats_and_cannot_become_training(tmp_path):
    from lucifer.behavioral_resources import attach_numeric_validation
    from lucifer.meditation_sources import _worker_dataset, prepare_meditation_sources, revalidate_meditation_sources
    from test_meditation_resume_scope import source
    from test_meditation_service import harness
    service, store, _ = harness(tmp_path)
    try:
        cid = await source(store, 'one-consented-chat')
        original = await prepare_meditation_sources(store, {'mode': 'supervised', 'conversation_ids': [cid]})
        bundle = attach_numeric_validation(original, 'precommitted-internal-validation')
        assert revalidate_meditation_sources(store, bundle)
        datasets = _worker_dataset(bundle)['datasets']
        assert datasets['train'] == _worker_dataset(original)['datasets']['train']
        assert len(datasets['development']) == len(datasets['holdout']) == 2
        assert len(bundle['manifest']['groups']) == 1
        assert bundle['automatic_validation']['training_eligible'] is False
        assert bundle['automatic_validation']['independent_human_source_groups_claimed'] is False
        assert datasets['development'] != datasets['holdout']
        changed = copy.deepcopy(bundle)
        changed['automatic_validation']['datasets']['development'][0]['messages'][-1]['content'] = '{"answer":999}'
        changed['automatic_validation']['sha256'] = digest({k: v for k, v in changed['automatic_validation'].items() if k != 'sha256'})
        changed['manifest']['automatic_validation_sha256'] = changed['automatic_validation']['sha256']
        changed['manifest']['manifest_sha256'] = digest({k: v for k, v in changed['manifest'].items() if k != 'manifest_sha256'})
        with pytest.raises(ValueError, match='oracle proof changed'):
            revalidate_meditation_sources(store, changed)
    finally:
        await service.close()
        store.close()
