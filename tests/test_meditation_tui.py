"""Meditation terminal contracts: real state, explicit activation, one heartbeat."""
import json
from uuid import UUID

import httpx
import pytest
from textual.widgets import Button, Select, Static

from lucifer.adapter_map import AdapterBrain, build_atlas
from lucifer.client import APIClient
from lucifer.tui import LuciferApp, MeditationReviewScreen, SleepProgressBar
from test_tui import Backend


class MeditationBackend(Backend):
    def __init__(self):
        super().__init__()
        self.meditations = []
        self.bases = []
        self.status.update(model=self.models[0], meditation_busy=False, learning_busy=False)

    def snapshot(self):
        result = super().snapshot()
        result['jobs']['meditation'] = self.meditations
        result['base_candidates'] = self.bases
        return result

    def handle(self, request):
        payload = json.loads(request.content) if request.content else {}
        path = request.url.path
        if path.startswith('/meditation'):
            self.requests.append((request.method, path, payload))
            if path == '/meditation/start':
                self.meditations.append({'id': 'meditation_one', 'status': 'queued', 'request': payload,
                    'progress': {'stage': 'queued', 'terminal': False, 'restored': False}})
                return httpx.Response(202, json=self.meditations[-1])
            if path == '/meditation/bases':
                return httpx.Response(200, json=self.bases)
            if path.endswith('/cancel'):
                self.meditations[-1].update(status='paused', resumable=True,
                    progress={'stage': 'paused', 'terminal': True, 'restored': True})
                self.status['meditation_busy'] = False
                return httpx.Response(200, json=self.meditations[-1])
            if path.endswith('/resume'):
                self.meditations[-1].update(status='queued', resumable=False,
                    progress={'stage': 'queued', 'terminal': False, 'restored': False})
                return httpx.Response(202, json=self.meditations[-1])
            if path.endswith('/activate'):
                return httpx.Response(200, json={'status': 'active', 'id': 'base_one'})
            if path.endswith('/rollback'):
                return httpx.Response(200, json={'status': 'restored'})
        return super().handle(request)


def test_cortex_hides_known_incompatible_experts_without_mutating_registry():
    models = [{'id': 'original', 'status': 'loaded'}]
    adapters = [{'id': 'old', 'base_model_id': 'original', 'status': 'archived', 'compatible': False},
                {'id': 'valid', 'base_model_id': 'original', 'status': 'admitted', 'compatible': True},
                {'id': 'legacy', 'base_model_id': 'original', 'status': 'candidate'}]
    atlas = build_atlas(models, adapters)
    assert atlas.node('adapter:old') is None
    assert atlas.node('adapter:valid') and atlas.node('adapter:legacy')
    assert len(adapters) == 3 and adapters[0]['compatible'] is False
    adapters[0]['compatible'] = True
    assert build_atlas(models, adapters).node('adapter:old') is not None


@pytest.mark.asyncio
async def test_meditation_client_preserves_retry_identity_and_scoped_payload():
    calls = []
    def handle(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={'id': 'one'})
    client = APIClient(transport=httpx.MockTransport(handle))
    try:
        await client.start_meditation(conversation_ids=['chat'], base_model='native', seed=7,
                                      dry_run=True, idempotency_key='start-intent')
        await client.meditation_action('one', 'resume', idempotency_key='resume-intent')
        await client.activate_base('new', idempotency_key='activate-intent')
        await client.rollback_base('parent', idempotency_key='rollback-intent')
        assert calls == [('/meditation/start', {'mode': 'supervised', 'dry_run': True, 'idempotency_key': 'start-intent',
                         'conversation_ids': ['chat'], 'base_model': 'native', 'seed': 7}),
                        ('/meditation/runs/one/resume', {'idempotency_key': 'resume-intent'}),
                        ('/meditation/bases/new/activate', {'idempotency_key': 'activate-intent'}),
                        ('/meditation/rollback', {'idempotency_key': 'rollback-intent', 'base_id': 'parent'})]
        with pytest.raises(ValueError, match='resume or cancel'):
            await client.meditation_action('one', 'invalid')
        with pytest.raises(ValueError, match='supervised or unsupervised'):
            await client.start_meditation(mode='automatic')
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_f6_native_meditation_uses_current_chat_scope_and_no_extra_polling():
    backend = MeditationBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view='meditation')
    async with app.run_test(size=(120, 42)) as pilot:
        assert app.active_view == 'meditation'
        assert app.query_one('#meditation-view').display
        assert not any(path == '/meditation/start' for _, path, _ in backend.requests)
        app.conversation = {'id': 'selected_chat', 'training_enabled': True, 'logging_enabled': True}
        app.query_one('#meditation-scope', Select).value = 'current'
        app.query_one('#meditation-mode', Select).value = 'unsupervised'
        await pilot.click('#meditation-start')
        await app.workers.wait_for_complete()
        await pilot.pause()
        starts = [data for _, path, data in backend.requests if path == '/meditation/start']
        assert len(starts) == 1 and starts[0]['conversation_ids'] == ['selected_chat']
        assert starts[0]['mode'] == 'unsupervised'
        UUID(starts[0]['idempotency_key'])
        assert app.conversation['training_enabled'] is True
        assert app.active_view == 'meditation'
        backend.status['meditation_busy'] = True
        backend.meditations[-1].update(status='training', controller={'learning_rate': 0.00002, 'microbatch': 16,
            'segment_updates': 25, 'successful_updates': 12}, progress={'stage': 'training', 'terminal': False})
        for _ in range(3):
            await app.refresh_panels()
        counters = app.query_one('#meditation-counters', Static).render().plain
        assert 'SUCCESSFUL UPDATES // 12' in counters and 'LEARNING RATE // 2e-05' in counters
        assert 'no global time' in counters
        assert app.query_one('#meditation-stage-bar', SleepProgressBar).total is None
        assert app.query_one('#meditation-start', Button).disabled
        assert not app.query_one('#meditation-cancel', Button).disabled
        assert sum(path == '/system/bootstrap' for _, path, _ in backend.requests) == 1
        assert not any(path in {'/models', '/adapters', '/meditation/runs', '/meditation/bases'} for _, path, _ in backend.requests)
        await pilot.click('#meditation-cancel')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.selected_meditation_job()['status'] == 'paused'
        assert not app.query_one('#meditation-resume', Button).disabled
        await pilot.press('f6')
        assert app.active_view == 'agent'
        await pilot.press('f6')
        assert app.active_view == 'meditation'
        await pilot.resize_terminal(80, 30)
        await pilot.pause()
        assert app.query_one('#meditation-review').region.width >= 9


@pytest.mark.asyncio
async def test_saved_base_requires_admission_and_rollback_before_explicit_activation():
    backend = MeditationBackend()
    backend.meditations = [{'id': 'meditation_one', 'status': 'admitted', 'saved_base_id': 'base_one',
                            'progress': {'stage': 'finished', 'terminal': True, 'restored': True}}]
    backend.bases = [{'id': 'base_one', 'model_id': 'native_one', 'status': 'admitted',
                     'admission_report': {'passed': True}, 'rollback_receipt': {'verified': False}}]
    backend.adapters = [{'id': 'wrong', 'base_model_id': 'qwen', 'status': 'archived',
                         'compatible': False, 'compatibility_reason': 'different base'},
                        {'id': 'right', 'base_model_id': 'demo', 'status': 'archived', 'compatible': True}]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view='meditation')
    async with app.run_test(size=(120, 42)) as pilot:
        assert app.query_one('#meditation-activate', Button).disabled
        assert app.query_one(AdapterBrain).atlas.node('adapter:wrong') is None
        assert all(value != 'wrong' for _, value in app._adapter_options)
        assert '1 INCOMPATIBLE HIDDEN' in app.query_one('#brain-heading', Static).render().plain
        assert len(app._adapters) == 2
        backend.bases[0]['rollback_receipt']['verified'] = True
        backend.revision += 1
        await app.refresh_panels()
        assert not app.query_one('#meditation-activate', Button).disabled
        assert not any(path.endswith('/activate') for _, path, _ in backend.requests)
        await pilot.click('#meditation-review')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, MeditationReviewScreen)
        await pilot.press('escape')
        assert not any(path.endswith('/activate') for _, path, _ in backend.requests)
        await pilot.click('#meditation-activate')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len([1 for _, path, _ in backend.requests if path == '/meditation/bases/base_one/activate']) == 1


@pytest.mark.asyncio
async def test_activation_rejection_is_visible_on_meditation_page_and_never_retried():
    backend = MeditationBackend()
    backend.meditations = [{'id': 'meditation_one', 'status': 'admitted', 'saved_base_id': 'base_one',
                            'progress': {'stage': 'finished', 'terminal': True, 'restored': True}}]
    backend.bases = [{'id': 'base_one', 'status': 'admitted', 'admission_report': {'passed': True},
                      'rollback_receipt': {'verified': True}}]
    attempts = []
    def handle(request):
        if request.url.path.endswith('/activate'):
            attempts.append(json.loads(request.content))
            return httpx.Response(409, json={'detail': 'Consumed source consent changed'})
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)), initial_view='meditation')
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click('#meditation-activate')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(attempts) == 1
        assert 'Consumed source consent changed' in app.query_one('#meditation-result', Static).render().plain
        assert not app.busy
        assert app.active_view == 'meditation'


@pytest.mark.asyncio
async def test_unadmitted_run_results_show_weight_proof_and_missing_evidence_without_activation():
    backend = MeditationBackend()
    backend.meditations = [{'id': 'meditation_one', 'status': 'awaiting_evidence', 'request': {'mode': 'unsupervised'},
        'progress': {'stage': 'awaiting_evidence', 'terminal': True, 'restored': True},
        'policy': {'successful_updates': 25},
        'training_precision': 'native_f32',
        'distribution_retention': {'passed': False, 'mean_forward_kl': 0.01},
        'weight_update_receipt': {'changed_elements': 65000000, 'delta_l2': 1.6},
        'native_update_receipt': {'passed': True, 'rollback_probe_passed': True},
        'admission_report': {'passed': False, 'status': 'awaiting_evidence', 'fresh_groups': 2, 'required_groups': 8}}]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view='meditation')
    async with app.run_test(size=(120, 42)) as pilot:
        assert not app.query_one('#meditation-review', Button).disabled
        assert app.query_one('#meditation-activate', Button).disabled
        assert 'FRESH HELDOUT GROUPS // 2 / 8' in app.query_one('#meditation-plan', Static).render().plain
        assert 'F32 compute / BF16 saved' in app.query_one('#meditation-plan', Static).render().plain
        await pilot.click('#meditation-review')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, MeditationReviewScreen)
        assert app.screen.record['weight_update_receipt']['changed_elements'] == 65000000
        assert app.screen.record['native_update_receipt']['rollback_probe_passed'] is True
        assert app.screen.record['distribution_retention']['passed'] is False
        assert not any(path.endswith('/activate') for _, path, _ in backend.requests)


@pytest.mark.asyncio
async def test_live_source_scope_shows_actual_selection_and_restores_next_start_choice():
    backend = MeditationBackend()
    backend.meditations = [{'id': 'selected_sources', 'status': 'running',
        'request': {'mode': 'supervised', 'conversation_ids': ['source_a', 'source_b']},
        'training_precision': 'native_f32', 'progress': {'stage': 'prepare', 'restored': False}}]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view='meditation')
    async with app.run_test(size=(120, 42)) as pilot:
        picker = app.query_one('#meditation-scope', Select)
        await pilot.pause()
        assert picker.disabled and picker.value == 'all'
        assert picker.query_one('#label', Static).render().plain == '2 selected training chats'
        plan = app.query_one('#meditation-plan', Static).render().plain
        assert 'SOURCES // 2 selected training chats' in plan and 'source_a' not in plan
        await app.command('/meditation sources')
        assert app.screen.record['conversation_ids'] == ['source_a', 'source_b']
        await app.pop_screen()
        backend.meditations[-1].update(status='paused', progress={'stage': 'paused', 'restored': True, 'terminal': True})
        await app.refresh_panels()
        assert not picker.disabled and picker.value == 'all'
        await pilot.click('#meditation-start')
        await app.workers.wait_for_complete()
        await pilot.pause()
        starts = [data for _, path, data in backend.requests if path == '/meditation/start']
        assert len(starts) == 1 and 'conversation_ids' not in starts[0]
