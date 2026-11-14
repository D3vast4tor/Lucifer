import httpx
import pytest
from textual.widgets import Static

from lucifer.client import APIClient
from lucifer.training_readings import LEGEND, outcome_lines, retained_updates
from lucifer.tui import LuciferApp
from test_meditation_tui import MeditationBackend


def test_readings_never_turn_missing_evidence_or_discarded_weights_into_success():
    assert 'Not measured yet' in '\n'.join(outcome_lines({}))
    job = {'checkpoint_path': 'best', 'weight_update_receipt': {'checkpoint_path': 'rejected', 'successful_updates': 57}}
    assert retained_updates(job) is None
    job['retained_checkpoint_updates'] = 25
    assert retained_updates(job) == 25
    assert 'not an IQ' not in '\n'.join(outcome_lines({}))
    assert 'Neither is an IQ score' in LEGEND


@pytest.mark.asyncio
async def test_both_pages_have_outcomes_and_responsive_explanation_without_extra_polling():
    backend = MeditationBackend()
    backend.meditations = [{'id': 'layout-fixture', 'status': 'awaiting_evidence', 'request': {},
        'progress': {'terminal': True, 'restored': True}, 'behavioral_report': {
            'categories': [{'category': 'novel_reasoning', 'baseline_solved': 1, 'candidate_solved': 2, 'total': 2}],
            'gains': 1, 'regressions': 0, 'complete': True, 'quality_status': 'not_improved'}}]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view='meditation')
    async with app.run_test(size=(190, 58)) as pilot:
        await pilot.pause()
        rendered = app.query_one('#meditation-outcomes', Static).render().plain
        assert 'Novel reasoning: 1/2 → 2/2' in rendered
        assert 'OUTCOME // NOT IMPROVED' in rendered
        assert app.query_one('#meditation-legend').parent.display
        assert not app.query_one('#meditation-legend-inline').display
        await pilot.resize_terminal(90, 50)
        await pilot.pause()
        assert not app.query_one('#meditation-legend').parent.display
        assert app.query_one('#meditation-legend-inline').display
        app.set_view('sleep')
        await pilot.pause()
        assert 'Not measured yet' in app.query_one('#sleep-outcomes', Static).render().plain
        assert app.query_one('#sleep-legend-inline').display


@pytest.mark.asyncio
async def test_terminal_resume_sends_explicit_complete_scope_with_retry_key():
    backend = MeditationBackend()
    backend.meditations = [{'id': 'one', 'status': 'paused', 'request': {'conversation_ids': ['old']},
        'resumable': True, 'progress': {'terminal': True, 'restored': True}}]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view='meditation')
    async with app.run_test(size=(120, 45)):
        await app.meditation_command(['resume', 'one', '--conversations', 'old,new', '--idempotency-key', 'scope-key'])
        calls = [request for request in backend.requests if request[1].endswith('/resume')]
        assert calls[-1] == ('POST', '/meditation/runs/one/resume',
                             {'idempotency_key': 'scope-key', 'conversation_ids': ['old', 'new']})
        with pytest.raises(ValueError, match='unique'):
            await app.meditation_command(['resume', 'one', '--conversations', 'same,same'])
