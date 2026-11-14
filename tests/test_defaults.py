from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.cli import attach_existing, startup_timeout
from lucifer.config import Settings


def registry(tmp_path):
    config = tmp_path / 'models.yaml'
    config.write_text('default_model: lucifer\nmodels:\n  - {id: lucifer, backend: demo, startup_timeout: 3600}\n')
    return Settings(data_dir=tmp_path / 'data', models_file=config)


def test_project_default_autoloads_and_public_alias_works(tmp_path):
    settings = registry(tmp_path)
    assert settings.default_model == 'lucifer'
    assert startup_timeout(settings, 'lucifer') == 3600
    with TestClient(create_app(settings)) as client:
        status = client.get('/system/status').json()
        assert status['model']['id'] == 'lucifer'
        assert status['model']['display_name'] == 'Lucifer'
        assert status['default_model'] == 'lucifer'
        response = client.post('/chat', json={'model': 'Lucifer', 'messages': [{'role': 'user', 'content': '2 + 2'}]})
        assert response.status_code == 200, response.text
        assert response.json()['model'] == 'Lucifer'
    # Explicit selection retains precedence over the project default.
    settings.models_file.write_text(settings.models_file.read_text() + '  - {id: override, backend: demo}\n')
    with TestClient(create_app(settings, autoload='override')) as client:
        assert client.get('/models/current').json()['id'] == 'override'


def test_existing_backend_must_match_project_before_opening_terminal(tmp_path, monkeypatch):
    from lucifer import cli
    from lucifer import tui
    settings = registry(tmp_path)
    calls = []
    status = {'version': cli.__version__, 'data_dir': str(settings.data_dir), 'models_file': str(settings.models_file), 'model': {'id': 'lucifer'}}
    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=status)
    original = httpx.Client
    monkeypatch.setattr(cli.httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    opened = []
    monkeypatch.setattr(tui, 'LuciferApp', lambda **kwargs: SimpleNamespace(run=lambda: opened.append(kwargs)))
    attach_existing('http://localhost:8000', settings, 'lucifer')
    assert len(opened) == 1 and calls == ['/system/status']
    status['models_file'] = str(Path('/different/project/models.yaml'))
    with pytest.raises(RuntimeError, match='different backend'):
        attach_existing('http://localhost:8000', settings, 'lucifer')
    assert len(opened) == 1


def test_doctor_resolves_profile_executables_relative_to_registry_outside_checkout(tmp_path, monkeypatch):
    import json
    from lucifer import cli

    configs = tmp_path / 'project' / 'configs'
    runtimes = tmp_path / 'project' / 'runtime'
    configs.mkdir(parents=True)
    runtimes.mkdir()
    for name in ('serve', 'vllm'):
        executable = runtimes / name
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o700)
    models_file = configs / 'models.yaml'
    models_file.write_text('''models:
  - id: primary
    backend: strata
    path: ../runtime/model
    runtime_command: [../runtime/serve]
  - id: train
    backend: vllm
    path: org/model
    executable: ../runtime/vllm
''')
    settings = Settings(data_dir=tmp_path / 'state', models_file=models_file)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    captured = []
    monkeypatch.setattr(cli, 'Console', lambda: SimpleNamespace(print_json=lambda value: captured.append(json.loads(value))))
    monkeypatch.setattr(cli.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=''))

    cli.doctor(settings)

    assert {model['id']: model['binary_available'] for model in captured[0]['models']} == {
        'primary': True, 'train': True,
    }


@pytest.mark.parametrize('load_status', [200, 409])
def test_attach_refreshes_stale_file_profile_with_absolute_paths_for_older_backend(tmp_path, monkeypatch, load_status):
    import json
    from lucifer import cli
    from lucifer.inference import ModelSpec

    project = tmp_path / 'relocated'
    configs = project / 'configs'
    configs.mkdir(parents=True)
    models = configs / 'models.yaml'
    models.write_text('''default_model: lucifer
models:
  - id: lucifer
    backend: strata
    path: ../var/strata/models/packs/primary
    runtime_command: [../scripts/serve-strata]
    startup_timeout: 3600
''')
    settings = Settings(data_dir=tmp_path / 'state', models_file=models, api_token='local-test-token')
    stale = ModelSpec(id='lucifer', backend='strata',
                      path='/old/project/var/strata/models/packs/primary',
                      runtime_command=['/old/project/var/strata/serve-lucifer.sh'])
    status = {'version': cli.__version__, 'data_dir': str(settings.data_dir),
              'models_file': str(settings.models_file), 'model': None,
              'agent_busy': False, 'learning_busy': False}
    loads, calls, opened = [], [], []

    def handle(request):
        assert request.headers['Authorization'] == 'Bearer local-test-token'
        calls.append((request.method, request.url.path))
        if request.url.path == '/system/status':
            return httpx.Response(200, json=status)
        if request.url.path == '/models':
            return httpx.Response(200, json=[{
                **stale.model_dump(), 'spec': stale.model_dump(),
                'registry_source': 'file', 'status': 'error'}])
        if request.url.path == '/models/load':
            body = json.loads(request.content)
            # A live backend with the old schema accepts no new load fields.
            assert set(body) <= {'model_id', 'spec'}
            assert body.get('model_id', 'lucifer') == 'lucifer'
            assert request.headers['X-Lucifer-Profile-Source'] == 'file'
            assert body['spec']['path'] == str(project / 'var/strata/models/packs/primary')
            assert body['spec']['runtime_command'] == [str(project / 'scripts/serve-strata')]
            assert request.extensions['timeout']['read'] >= 3600
            loads.append(body)
            return httpx.Response(load_status, json={**body['spec'], 'status': 'loaded',
                                                   'detail': 'An agent run is active'})
        raise AssertionError(f'Unexpected request: {request.method} {request.url.path}')

    original = httpx.Client
    monkeypatch.setattr(cli.httpx, 'Client', lambda **kwargs: original(
        transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(cli, 'run_tui', lambda *args, **kwargs: opened.append(args))
    monkeypatch.chdir(project)

    if load_status == 200:
        attach_existing('http://localhost:8000', settings, 'lucifer')
    else:
        with pytest.raises(RuntimeError, match='Could not select the model'):
            attach_existing('http://localhost:8000', settings, 'lucifer')

    assert len(loads) == 1 and len(opened) == (1 if load_status == 200 else 0)
    assert calls == [('GET', '/system/status'), ('GET', '/models'), ('POST', '/models/load')]


@pytest.mark.parametrize('source', ['runtime', 'persisted'])
def test_attach_keeps_explicit_registered_profile_instead_of_replacing_with_file(tmp_path, monkeypatch, source):
    import json
    from lucifer import cli
    from lucifer.inference import ModelSpec

    settings = registry(tmp_path)
    custom = ModelSpec(id='lucifer', backend='external', endpoint='http://127.0.0.1:8901')
    status = {'version': cli.__version__, 'data_dir': str(settings.data_dir),
              'models_file': str(settings.models_file), 'model': None}
    loads, opened = [], []

    def handle(request):
        if request.url.path == '/system/status':
            return httpx.Response(200, json=status)
        if request.url.path == '/models':
            return httpx.Response(200, json=[{
                **custom.model_dump(), 'spec': custom.model_dump(),
                'registry_source': source, 'status': 'ready'}])
        assert request.url.path == '/models/load'
        body = json.loads(request.content)
        assert body == {'model_id': 'lucifer'}
        loads.append(body)
        return httpx.Response(200, json={**custom.model_dump(), 'status': 'loaded'})

    original = httpx.Client
    monkeypatch.setattr(cli.httpx, 'Client', lambda **kwargs: original(
        transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(cli, 'run_tui', lambda *args, **kwargs: opened.append(args))

    attach_existing('http://localhost:8000', settings, 'lucifer')

    assert len(loads) == 1 and len(opened) == 1


def test_attach_keeps_file_profile_missing_from_current_configuration(tmp_path, monkeypatch):
    import json
    from lucifer import cli
    from lucifer.inference import ModelSpec

    settings = registry(tmp_path)
    saved = ModelSpec(id='retained', backend='demo')
    status = {'version': cli.__version__, 'data_dir': str(settings.data_dir),
              'models_file': str(settings.models_file), 'model': None}
    loads, opened = [], []

    def handle(request):
        if request.url.path == '/system/status':
            return httpx.Response(200, json=status)
        if request.url.path == '/models':
            return httpx.Response(200, json=[{
                **saved.model_dump(), 'spec': saved.model_dump(),
                'registry_source': 'file', 'status': 'ready'}])
        assert request.url.path == '/models/load'
        body = json.loads(request.content)
        assert body == {'model_id': 'retained'}
        loads.append(body)
        return httpx.Response(200, json={**saved.model_dump(), 'status': 'loaded'})

    original = httpx.Client
    monkeypatch.setattr(cli.httpx, 'Client', lambda **kwargs: original(
        transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(cli, 'run_tui', lambda *args, **kwargs: opened.append(args))

    attach_existing('http://localhost:8000', settings, 'retained')

    assert len(loads) == 1 and len(opened) == 1
