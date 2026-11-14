"""Native vLLM chat counting uses cached startup routes and shared pair counts."""
import json

import httpx
import pytest
import yaml

from lucifer.capabilities import CapabilityRegistry
from lucifer.config import Settings
from lucifer.inference import ModelManager
from lucifer.learning import LearningService
from lucifer.providers import ProviderError, conservative_count
from lucifer.store import Store


async def native_engine(tmp_path, *, advertised=True, reply=None, prefix='', history_contract=False):
    path = tmp_path / 'models.yaml'
    path.write_text(yaml.safe_dump({'models': [{'id': 'trainable', 'backend': 'vllm', 'path': '/local/checkpoint',
        'endpoint': 'http://localhost:8091' + prefix + '/v1', 'served_model_name': 'Lucifer', 'context_length': 2048,
        'capabilities': {'reasoning': {'constraints': {'history_field': 'reasoning_content'}}} if history_contract else {},
        'chat_template_kwargs': {'custom_rule': 'kept'}}]}))
    settings = Settings(data_dir=tmp_path / 'data', models_file=path)
    engine = ModelManager(settings, Store(settings.data_dir / 'state.db'))
    requests = []
    def handle(request):
        payload = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, payload))
        path = request.url.path[len(prefix):] if prefix and request.url.path.startswith(prefix) else request.url.path
        if path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'Lucifer', 'max_model_len': 2048}]})
        if path == '/v1/status':
            return httpx.Response(404)
        if path == '/openapi.json':
            paths = {'/v1/chat/completions': {'post': {}}}
            if advertised:
                paths['/tokenize'] = {'post': {}}
            return httpx.Response(200, json={'paths': paths})
        if path == '/tokenize':
            return reply(request) if reply else httpx.Response(200, json={'count': 300, 'tokens': list(range(300)), 'max_model_len': 2048})
        assert path == '/v1/chat/completions'
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=
            'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    await engine.client.aclose()
    engine.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    await engine.load('trainable')
    return engine, requests


@pytest.mark.asyncio
async def test_exact_native_chat_count_fits_real_prompt_with_identical_template_options(tmp_path):
    engine, requests = await native_engine(tmp_path)
    messages = [{'role': 'user', 'content': 'A finite source replay trace. ' * 40}]
    try:
        snapshot = engine.capabilities.get('trainable')
        assert conservative_count(messages, [])['input_tokens'] + 256 > 2048
        with pytest.raises(ProviderError, match='exceeds active context'):
            _ = [item async for item in engine.agent_events(messages, model_id='trainable', max_tokens=256, thinking='off')]
        count = await engine.count_tokens(messages, model_id='trainable', profile_revision=snapshot.profile_revision,
                                         max_tokens=256, thinking='off')
        assert count == {'input_tokens': 300, 'method': 'vllm_tokenize_chat', 'estimated': False}
        result = [item async for item in engine.agent_events(messages, model_id='trainable', max_tokens=256,
                  thinking='off', tool_choice='none', input_tokens=count['input_tokens'])]
        assert ('delta', {'content': '{}'}) in result
        token = next(payload for method, path, payload in requests if method == 'POST' and path == '/tokenize')
        chat = next(payload for method, path, payload in requests if method == 'POST' and path == '/v1/chat/completions')
        assert token['model'] == chat['model'] == 'Lucifer' and token['messages'] == chat['messages'] == messages
        assert token['chat_template_kwargs'] == {**chat['chat_template_kwargs'], 'reasoning_effort': chat['reasoning_effort']}
        assert token['chat_template_kwargs']['enable_thinking'] is False
        assert token['chat_template_kwargs']['custom_rule'] == 'kept'
        assert token['add_generation_prompt'] is True and token['continue_final_message'] is False
        assert token['add_special_tokens'] is False and 'tools' not in token and 'response_format' not in token
        assert engine.capabilities.get('trainable') is snapshot
        assert sum(path == '/openapi.json' for _, path, _ in requests) == 1
        assert sum(path == '/tokenize' for _, path, _ in requests) == 1
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_unknown_tokenizer_route_is_not_probed_or_claimed_exact(tmp_path):
    engine, requests = await native_engine(tmp_path, advertised=False)
    try:
        count = await engine.count_tokens([{'role': 'user', 'content': 'Question'}], model_id='trainable')
        assert count['estimated'] is True and count['input_tokens'] > 0
        assert not any(path == '/tokenize' for _, path, _ in requests)
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_native_tokenizer_preserves_reverse_proxy_api_prefix(tmp_path):
    engine, requests = await native_engine(tmp_path, prefix='/proxy')
    try:
        count = await engine.count_tokens([{'role': 'user', 'content': 'Question'}], model_id='trainable')
        assert count['estimated'] is False
        assert any(path == '/proxy/tokenize' for _, path, _ in requests)
        assert not any(path == '/tokenize' for _, path, _ in requests)
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.parametrize('declared', [False, True])
@pytest.mark.asyncio
async def test_count_and_generation_share_only_declared_native_reasoning_history(tmp_path, declared):
    engine, requests = await native_engine(tmp_path, history_contract=declared)
    messages = [{'role': 'user', 'content': 'First question'},
                {'role': 'assistant', 'content': 'First answer', 'reasoning_content': 'ACTUAL_RETAINED_THOUGHT'},
                {'role': 'user', 'content': 'Follow-up'}]
    try:
        count = await engine.count_tokens(messages, model_id='trainable', thinking='off', max_tokens=256)
        _ = [item async for item in engine.agent_events(messages, model_id='trainable', thinking='off', max_tokens=256,
                                                       input_tokens=count['input_tokens'])]
        token = next(body for method, path, body in requests if path == '/tokenize')
        chat = next(body for method, path, body in requests if path == '/v1/chat/completions')
        assert token['messages'] == chat['messages']
        assert ('reasoning_content' in chat['messages'][1]) is declared
        assert messages[1]['reasoning_content'] == 'ACTUAL_RETAINED_THOUGHT'  # RAM source is unmodified.
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.parametrize('bad', [{'count': 0, 'tokens': []}, {'count': True, 'tokens': [1]}, {'count': 2, 'tokens': [1]},
                                {'count': 1, 'tokens': [-1]}, {'count': 1}, {'count': '1', 'tokens': [1]}])
@pytest.mark.asyncio
async def test_invalid_native_counts_keep_safe_positive_estimate(tmp_path, bad):
    engine, _ = await native_engine(tmp_path, reply=lambda request: httpx.Response(200, json=bad))
    try:
        count = await engine.count_tokens([{'role': 'user', 'content': 'Question'}], model_id='trainable')
        assert count['estimated'] is True and count['method'] == 'conservative_estimate'
        assert count['input_tokens'] > 0 and count['count_error'] == 'native_tokenizer_unavailable'
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_unavailable_native_count_keeps_safe_estimate_without_changing_capabilities(tmp_path):
    engine, requests = await native_engine(tmp_path, reply=lambda request: httpx.Response(503))
    try:
        snapshot = engine.capabilities.get('trainable')
        count = await engine.count_tokens([{'role': 'user', 'content': 'Question'}], model_id='trainable')
        assert count['estimated'] is True and count['input_tokens'] > 0
        assert engine.capabilities.get('trainable') is snapshot
        assert sum(path == '/tokenize' for _, path, _ in requests) == 1
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_native_count_does_not_claim_unsupported_template_modes(tmp_path):
    engine, requests = await native_engine(tmp_path)
    try:
        messages = [{'role': 'user', 'content': 'Question'}]
        tool = {'type': 'function', 'function': {'name': 'example', 'parameters': {'type': 'object', 'properties': {}}}}
        for kwargs in ({'thinking': 'high'}, {'tools': [tool]}, {'response_format': {'type': 'json_object'}}):
            count = await engine.count_tokens(messages, model_id='trainable', **kwargs)
            assert count['estimated'] is True
        assert not any(path == '/tokenize' for _, path, _ in requests)
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_evaluation_counts_each_case_once_and_shares_honest_count_between_pair(tmp_path):
    settings = Settings(data_dir=tmp_path / 'data', models_file=tmp_path / 'models.yaml', eval_file=tmp_path / 'evals.yaml')
    settings.eval_file.write_text(json.dumps({'cases': [
        {'id': 'task', 'prompt': 'long formal task', 'expected_exact': '{}'},
        {'id': 'safety', 'suite': 'safety', 'prompt': 'privacy', 'expected_exact': '{}'}]}))
    store = Store(settings.data_dir / 'state.db')
    class Manager:
        def __init__(self):
            self.capabilities = CapabilityRegistry()
            self.capabilities.record('base', 'revision', declared={'chat_completions': True})
            self.count_calls, self.turns = [], []
        def current(self):
            return {'id': 'base', 'backend': 'vllm', 'effective_context_length': 2048}
        async def count_tokens(self, messages, tools, **kwargs):
            self.count_calls.append((messages, tools, kwargs))
            return {'input_tokens': 300 if messages[0]['content'] == 'long formal task' else 20,
                    'method': 'vllm_tokenize_chat', 'estimated': False}
        async def agent_events(self, messages, tools, **kwargs):
            assert kwargs['input_tokens'] == (300 if messages[0]['content'] == 'long formal task' else 20)
            assert kwargs['thinking'] == 'off' and kwargs['tool_choice'] == 'none' and tools == []
            self.turns.append(kwargs)
            yield 'delta', {'content': '{}'}
            yield 'finish', {'finish_reason': 'stop'}
        async def activate_adapter(self, adapter):
            pass
    manager = Manager()
    service = LearningService(settings, store, manager)
    store.put('adapters', {'id': 'candidate', 'base_model_id': 'base', 'status': 'candidate'})
    try:
        evaluation = await service.evaluate('candidate')
        assert len(manager.count_calls) == 2 and len(manager.turns) == 4
        assert evaluation['metrics']['unscored_pairs'] == 0
        for case in evaluation['cases']:
            for side in ('baseline', 'candidate'):
                receipt = case[side + '_completion']
                assert receipt['token_count_method'] == 'vllm_tokenize_chat' and receipt['token_count_estimated'] is False
                assert receipt['input_tokens'] == (300 if case['id'] == 'task' else 20)
    finally:
        await service.close()
        store.close()
