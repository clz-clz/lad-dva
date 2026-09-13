import asyncio
import json

import pytest

from live_backbone import LiveBackboneSettings
from official_contract import official_manifest_decoder_constants


@pytest.mark.parametrize('seconds', [120, 300, 600])
def test_qwen_timeout_env_is_applied(monkeypatch, seconds):
    monkeypatch.setenv('BACKBONE_PROVIDER', 'vllm')
    monkeypatch.setenv('BACKBONE_MODEL', 'Qwen/Qwen3-32B-AWQ')
    monkeypatch.setenv('BACKBONE_REVISION', '0499c3ac83fdef8810b907a23894ba91e95eddd8')
    monkeypatch.setenv('QWEN_PROVIDER_TIMEOUT_SECONDS', str(seconds))
    assert LiveBackboneSettings.from_env().timeout_seconds == seconds


def test_manifest_records_actual_timeout():
    result = official_manifest_decoder_constants(3600, provider_timeout=300)
    assert result['provider_timeout_seconds'] == 300
    assert result['runner_request_timeout_seconds'] == 3600


def test_qwen_override_does_not_change_deepseek(monkeypatch):
    monkeypatch.setenv('BACKBONE_PROVIDER', 'deepseek')
    monkeypatch.setenv('BACKBONE_MODEL', 'deepseek-v4-flash')
    monkeypatch.setenv('QWEN_PROVIDER_TIMEOUT_SECONDS', '300')
    assert LiveBackboneSettings.from_env().timeout_seconds == 120


def test_log_failure_preserves_provider_exception(monkeypatch, tmp_path):
    import request_diagnostics as diagnostics
    monkeypatch.setenv('QWEN_DIAGNOSTICS_DIR', str(tmp_path))
    original = TimeoutError('provider')
    class Adapter:
        @diagnostics.traced_request
        def structured_requester(self, stage, payload):
            raise original
    def broken(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(diagnostics, 'emit', broken)
    monkeypatch.setattr(diagnostics.logging, 'error', broken)
    with pytest.raises(TimeoutError) as caught:
        Adapter().structured_requester('coder', {})
    assert caught.value is original


@pytest.mark.parametrize('broken_logging', [False, True])
def test_http_evidence_never_turns_received_response_into_retry(monkeypatch, tmp_path, broken_logging):
    import httpx
    from openai import OpenAI
    import request_diagnostics as diagnostics
    monkeypatch.setenv('QWEN_DIAGNOSTICS_DIR', str(tmp_path))
    calls = []
    def respond(self, request):
        calls.append(request)
        return httpx.Response(200, request=request, json={
            'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': 'test', 'choices': None, 'usage': None})
    def broken(*args, **kwargs):
        raise OSError('sink unavailable')
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', respond)
    if broken_logging:
        monkeypatch.setattr(diagnostics, 'emit', broken)
        monkeypatch.setattr(diagnostics.logging, 'error', broken)
    with OpenAI(api_key='test', base_url='http://localhost/v1',
                http_client=diagnostics.diagnostic_http_client()) as client:
        result = client.chat.completions.create(model='test', messages=[])
    assert result.choices is None
    assert len(calls) == 1


@pytest.mark.parametrize('seconds', [0, -1, 121, float('nan'), float('inf')])
def test_invalid_qwen_timeout_rejected(seconds):
    with pytest.raises(ValueError):
        LiveBackboneSettings('vllm', 'Qwen/Qwen3-32B-AWQ', 'http://localhost/v1',
                             'key', revision='a' * 40, timeout_seconds=seconds)


def test_trace_context_survives_threads_and_retains_failure(tmp_path, monkeypatch):
    from request_diagnostics import sentence_context, traced_request, queue_context
    monkeypatch.setenv('QWEN_DIAGNOSTICS_DIR', str(tmp_path))
    class Adapter:
        @traced_request
        def structured_requester(self, stage, payload):
            raise TimeoutError('secret must not be logged')
    async def call(index):
        with sentence_context(dataset='msra', noise='BT', seed=13, row_index=index):
            with queue_context(index / 10), pytest.raises(TimeoutError):
                await asyncio.to_thread(Adapter().structured_requester, 'coder',
                                        {'name': 'path_1', 'messages': [], 'schema': {}})
    async def main():
        await asyncio.gather(call(7), call(9))
    asyncio.run(main())
    events = [json.loads(s) for s in (tmp_path / 'events.jsonl').read_text().splitlines()]
    failed = [r for r in events if r['event'] == 'request_failed']
    assert {r['row_index'] for r in failed} == {7, 9}
    assert all(r['exception_chain'] == ['TimeoutError'] for r in failed)
    starts = [r for r in events if r['event'] == 'request_started']
    assert all(r['queue_seconds'] == r['row_index'] / 10 for r in starts)
    assert 'secret must not be logged' not in (tmp_path / 'events.jsonl').read_text()


def test_real_sdk_retries_keep_request_id_and_hide_authorization(tmp_path, monkeypatch):
    import httpx
    from live_backbone import OpenAICompatibleLADRGAdapter
    from request_diagnostics import sentence_context
    monkeypatch.setenv('QWEN_DIAGNOSTICS_DIR', str(tmp_path))
    calls = []
    model = 'Qwen/Qwen3-32B-AWQ'
    revision = '0499c3ac83fdef8810b907a23894ba91e95eddd8'
    def respond(self, request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout('private credential string', request=request)
        return httpx.Response(200, request=request, json={
            'id':'test', 'object':'chat.completion', 'created':0,
            'model':model + '@' + revision,
            'choices':[{'index':0, 'finish_reason':'stop',
                        'message':{'role':'assistant', 'content':'{"tags":["O"]}'}}],
            'usage':{'prompt_tokens':17,'completion_tokens':3,'total_tokens':20,
                     'completion_tokens_details':{'reasoning_tokens':2}}})
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', respond)
    adapter = OpenAICompatibleLADRGAdapter(LiveBackboneSettings(
        'vllm',model,'http://localhost/v1','private credential string',
        revision=revision, timeout_seconds=300))
    try:
        with sentence_context(dataset='msra',noise='BT',seed=13,row_index=11):
            result = adapter.structured_requester('coder', {
                'name':'coder_path_1','schema':{'type':'object','properties':{
                    'tags':{'type':'array','items':{'type':'string'}}}},
                'messages':[{'role':'user','content':'repair'}],
                'temperature':1.0,'enable_thinking':True})
        assert result['tags'] == ['O']
        assert calls[0].extensions['timeout']['read'] == 300
    finally:
        adapter.close()
    text = (tmp_path / 'events.jsonl').read_text()
    events = [json.loads(line) for line in text.splitlines()]
    attempts = [e for e in events if e['event'] == 'http_started']
    assert [e['retry_index'] for e in attempts] == [0,1]
    assert len({e['request_id'] for e in events}) == 1
    assert all(e['row_index'] == 11 for e in events)
    assert 'private credential string' not in text
    completed = next(e for e in events if e['event'] == 'http_completed')
    assert completed['usage']['completion_tokens_details']['reasoning_tokens'] == 2


def test_request_evidence_selection_requires_exact_directory_and_full_id(tmp_path):
    import request_diagnostics as diagnostics

    first = 'a' * 32
    second = 'b' * 32
    (tmp_path / f'{first}.json').write_text('{"name":"first"}', encoding='utf-8')
    (tmp_path / f'{second}.json').write_text('{"name":"second"}', encoding='utf-8')
    (tmp_path / 'events.jsonl').write_text(
        json.dumps({'request_id': first, 'event': 'request_started'}) + '\n'
        + json.dumps({'request_id': second, 'event': 'request_started'}) + '\n',
        encoding='utf-8',
    )

    selected = diagnostics.select_request_evidence(tmp_path, first)
    assert selected['request_path'].name == f'{first}.json'
    assert selected['stream_path'] is None
    assert [event['request_id'] for event in selected['events']] == [first]
    with pytest.raises(ValueError, match='full 32-character'):
        diagnostics.select_request_evidence(tmp_path, first[:8])


def test_stream_diagnostics_incrementally_preserve_content_before_cutoff(
    tmp_path, monkeypatch,
):
    import httpx
    import request_diagnostics as diagnostics

    request_id = 'c' * 32
    monkeypatch.setenv('QWEN_DIAGNOSTICS_DIR', str(tmp_path))

    class FailingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'content-before-cutoff'
            raise TimeoutError('provider cutoff')

        def close(self):
            pass

    def respond(_transport, request):
        return httpx.Response(200, request=request, stream=FailingStream())

    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', respond)
    request_token = diagnostics._request.set({
        'request_id': request_id, 'stage': 'coder', 'path_name': 'path_5',
        'prompt_sha256': 'd' * 64, 'enable_thinking': False,
        'thinking_mode': 'nothink',
    })
    try:
        client = diagnostics.diagnostic_http_client()
        try:
            response = client.send(
                httpx.Request('GET', 'http://localhost/v1/chat/completions'),
                stream=True,
            )
            with pytest.raises(TimeoutError, match='provider cutoff'):
                b''.join(response.iter_bytes())
            response.close()
        finally:
            client.close()
    finally:
        diagnostics._request.reset(request_token)

    assert (tmp_path / f'{request_id}.stream.bin').read_bytes() == b'content-before-cutoff'
    selected = diagnostics.select_request_evidence(tmp_path, request_id)
    assert selected['stream_path'].name == f'{request_id}.stream.bin'
    assert any(event['event'] == 'http_stream_failed' for event in selected['events'])
