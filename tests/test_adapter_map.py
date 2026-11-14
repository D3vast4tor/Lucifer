import pytest
from textual.app import App, ComposeResult

from lucifer.adapter_map import AdapterBrain, Atlas, BrailleCanvas, build_atlas, node_details


MODELS = [{"id": "lucifer", "status": "loaded", "native_context_length": 1048576,
           "effective_context_length": 1048576}]
ADAPTERS = [
    {"id": "trained-one", "base_model_id": "lucifer", "status": "archived", "metrics": {"passed": True}},
    {"id": "trained-two", "base_model_id": "lucifer", "parent_adapter_id": "trained-one", "status": "active", "rank": 16},
    {"id": "unproven", "base_model_id": "lucifer", "status": "candidate"},
]


def test_topology_uses_real_lineage_and_does_not_fabricate_adapters():
    atlas = build_atlas(MODELS, ADAPTERS, active_adapter="trained-two")
    assert len(atlas.nodes) == 4
    assert {node.data.get("id") for node in atlas.nodes if node.kind == "adapter"} == {a["id"] for a in ADAPTERS}
    assert any(edge.source == "adapter:trained-one" and edge.target == "adapter:trained-two" and edge.kind == "lineage" for edge in atlas.edges)
    assert atlas.node("adapter:trained-two").status == "active"
    assert "1,048,576" in node_details(atlas.node("model:lucifer"), atlas)
    empty = build_atlas(MODELS, [])
    assert len(empty.nodes) == 1 and empty.edges == ()
    assert "navigation surface" in node_details(empty.nodes[0], empty)
    assert "No registry nodes" in node_details(None, Atlas((), (), None))


def test_unknown_parent_does_not_create_a_fake_adapter_and_source_job_is_exact():
    adapters = [{"id": "candidate", "base_model_id": "missing-base", "parent_adapter_id": "missing-parent", "status": "candidate"}]
    jobs = {"sleep": [{"id": "wrong", "candidate_adapter_id": "different", "total_examples": 999},
                      {"id": "real", "candidate_adapter_id": "candidate", "total_examples": 8}]}
    atlas = build_atlas([], adapters, jobs=jobs)
    assert len(atlas.nodes) == 2
    assert atlas.node("adapter:missing-parent") is None
    assert atlas.node("model:missing-base").status == "unavailable"
    selected = atlas.node("adapter:candidate")
    assert selected.data["source_job"]["id"] == "real"
    assert "8" in node_details(selected, atlas) and "999" not in node_details(selected, atlas)


class MapApp(App):
    def compose(self) -> ComposeResult:
        yield AdapterBrain(id="map")


@pytest.mark.asyncio
async def test_braille_cortex_rotation_zoom_selection_and_memoized_data():
    app = MapApp()
    async with app.run_test(size=(96, 32)) as pilot:
        brain = app.query_one(AdapterBrain)
        brain.set_data(MODELS, ADAPTERS)
        await pilot.pause()
        first = brain.render().plain
        assert any(0x2800 <= ord(c) <= 0x28ff for c in first)
        assert "CORE" in first and "ADAPTER ATLAS" in first
        assert len(first.splitlines()) == brain.content_size.height
        revision = brain.data_revision
        signature = brain.snapshot_signature()
        brain.set_data(MODELS, ADAPTERS)
        assert brain.data_revision == revision and brain.snapshot_signature() == signature
        assert brain.render() is brain.render()
        brain.focus()
        await pilot.press("right", "up", "+")
        assert brain.yaw > .1 and brain.pitch < .82 and brain.zoom_level > 1
        assert brain.render().plain != first
        selected = brain.selected_id
        await pilot.press("n")
        assert brain.selected_id != selected
        before = brain.pan_x
        await pilot.press("shift+right")
        assert brain.pan_x > before
        brain.action_reset()
        assert brain.yaw == .1 and brain.zoom_level == 1
        brain.render()
        ident, (x, y) = next(iter(brain._projected.items()))
        await pilot.click("#map", offset=(x, y))
        assert brain.selected_id == ident
        brain.set_data(MODELS, [])
        await pilot.pause()
        assert "NO ADAPTERS / BASE CORE ONLY" in brain.render().plain


def test_consolidated_expert_has_all_recorded_parents_and_topics():
    adapters = [*ADAPTERS, {'id': 'consolidated', 'base_model_id': 'lucifer', 'status': 'candidate',
        'source_expert_ids': ['trained-one', 'trained-two', 'missing'], 'parent_adapter_id': 'trained-one',
        'expert_name': 'Lucifer signal theory', 'expert_topics': ['signals', 'algebra']}]
    atlas = build_atlas(MODELS, adapters)
    sources = [edge.source for edge in atlas.edges if edge.target == 'adapter:consolidated']
    assert set(sources) == {'adapter:trained-one', 'adapter:trained-two'} and len(sources) == 2
    details = node_details(atlas.node('adapter:consolidated'), atlas)
    assert 'Lucifer signal theory' in details and 'signals, algebra' in details
    assert atlas.node('adapter:missing') is None


def test_clipped_lines_cover_the_visible_crossing_and_batch_identical_styles():
    canvas = BrailleCanvas(24, 8)
    canvas.line((-10**9, 2), (10**9, 2), "cyan")
    assert len(canvas.cells) == 24
    canvas.line((-100, -100), (-20, -20), "red")
    assert len(canvas.cells) == 24
    canvas.label(2, 1, "E01", "bold yellow")
    result = canvas.render()
    assert result.plain.splitlines()[0] == chr(0x2800 + 4 + 32) * 24
    assert result.plain.splitlines()[1][2:5] == "E01"
    assert len(result.spans) == 2  # One visible line and one label, not 192 cell spans.


@pytest.mark.asyncio
async def test_free_three_axis_rotation_drag_and_new_expert_reveal():
    app = MapApp()
    async with app.run_test(size=(96, 32)) as pilot:
        brain = app.query_one(AdapterBrain)
        brain.set_data(MODELS, ADAPTERS)
        await pilot.pause()
        brain.action_rotate(0, 20)
        assert abs(brain.pitch) > 1.2  # Full pitch instead of the old hard stop.
        before = brain.render().plain
        brain.focus()
        await pilot.press("e")
        assert brain.roll_angle > 0 and brain.render().plain != before
        yaw, pitch = brain.yaw, brain.pitch
        await pilot.mouse_down("#map", offset=(20, 10))
        await pilot.hover("#map", offset=(30, 14))
        await pilot.mouse_up("#map", offset=(30, 14))
        assert brain.yaw != yaw and brain.pitch != pitch
        assert app.mouse_captured is None
        expert = {"id": "new-real-expert", "base_model_id": "lucifer", "status": "candidate",
                  "is_expert": True, "expert_name": "Lucifer new expert", "expert_training_status": "trained"}
        brain.set_data(MODELS, [*ADAPTERS, expert])
        assert brain.atlas.node("adapter:new-real-expert").label.startswith("E")
        brain.pan_x, brain.pan_y = 10, -10
        brain.render()
        angles = brain.yaw, brain.pitch, brain.roll_angle
        assert brain.select_node("adapter:new-real-expert", reveal=True)
        brain.render()
        x, y = brain._projected["adapter:new-real-expert"]
        assert abs(x - brain.content_size.width / 2) <= 1
        assert abs(y - brain.content_size.height / 2) <= 1
        assert (brain.yaw, brain.pitch, brain.roll_angle) == angles
        assert "Lucifer new expert" in node_details(brain.selected(), brain.atlas)
        brain.action_reset()
        assert brain.roll_angle == 0 and brain.pitch == .82
