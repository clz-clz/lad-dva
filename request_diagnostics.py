"""Opt-in, credential-free request evidence independent of atomic result cells."""
import contextlib
import contextvars
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid
from collections.abc import Mapping

_sentence = contextvars.ContextVar('diagnostic_sentence', default={})
_request = contextvars.ContextVar('diagnostic_request', default={})
_queue = contextvars.ContextVar('diagnostic_queue_seconds', default=None)
_lock = threading.Lock()
_REQUEST_ID_RE = re.compile(r'^[0-9a-f]{32}$')
_FORBIDDEN_LABEL_KEYS = frozenset({'gold', 'gold_tags', 'gold_labels', 'ner_tags', 'target_tags'})


@contextlib.contextmanager
def queue_context(seconds):
    token = _queue.set(seconds)
    try:
        yield
    finally:
        _queue.reset(token)


@contextlib.contextmanager
def sentence_context(**identity):
    token = _sentence.set(identity)
    try:
        yield
    finally:
        _sentence.reset(token)


def exception_chain(exc):
    result, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        result.append(type(exc).__name__)
        exc = exc.__cause__ or exc.__context__
    return result


def emit(event, **fields):
    root = os.environ.get('QWEN_DIAGNOSTICS_DIR')
    if not root:
        return
    record = {**_sentence.get(), **_request.get(), 'event': event,
              'timestamp': time.time(), **fields}
    with _lock:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()


def _report_sink_failure(exc):
    try:
        logging.error('Request diagnostic sink failed (%s)', type(exc).__name__)
    except Exception:
        pass  # Even a failing logging handler must not affect provider semantics.


def safe_emit(event, **fields):
    try:
        emit(event, **fields)
    except Exception as exc:
        _report_sink_failure(exc)


def _redact_target_labels(value):
    """Remove target-label fields before model-facing evidence is persisted."""
    if isinstance(value, dict):
        return {
            key: _redact_target_labels(child)
            for key, child in value.items()
            if str(key).lower() not in _FORBIDDEN_LABEL_KEYS
        }
    if isinstance(value, list):
        return [_redact_target_labels(child) for child in value]
    if isinstance(value, tuple):
        return [_redact_target_labels(child) for child in value]
    return value


def _effective_thinking(self, requested):
    resolver = getattr(getattr(self, 'settings', None), 'effective_enable_thinking', None)
    if callable(resolver) and type(requested) is bool:
        try:
            return resolver(requested)
        except Exception:
            # The adapter remains the source of validation errors. Diagnostics
            # must never replace a provider or schema exception.
            return requested
    return requested


def _thinking_mode(enabled):
    if enabled is True:
        return 'thinking'
    if enabled is False:
        return 'nothink'
    return 'request-controlled'


def select_request_evidence(root, request_id):
    """Select exactly one diagnostic request by directory and full request ID.

    Prefixes and globs are deliberately rejected so a diagnosis cannot
    accidentally combine neighboring requests.  The return value contains
    paths only when they exist and events whose request_id is an exact match.
    """
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
        raise ValueError('request_id must be the full 32-character lowercase hex ID')
    directory = Path(root)
    if not directory.exists() or not directory.is_dir():
        raise ValueError('diagnostic root must be an existing directory')
    request_path = directory / f'{request_id}.json'
    stream_path = directory / f'{request_id}.stream.bin'
    events_path = directory / 'events.jsonl'
    events = []
    if events_path.exists():
        for line in events_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('request_id') == request_id:
                events.append(event)
    return {
        'request_id': request_id,
        'request_path': request_path if request_path.exists() else None,
        'stream_path': stream_path if stream_path.exists() else None,
        'events': events,
    }


def traced_request(fn):
    @functools.wraps(fn)
    def invoke(self, stage, payload):
        if not os.environ.get('QWEN_DIAGNOSTICS_DIR'):
            return fn(self, stage, payload)
        # Only model-facing fields; never serialize credentials or target labels.
        source = payload if isinstance(payload, Mapping) else {}
        safe = _redact_target_labels({key: source[key] for key in
                ('name', 'schema', 'messages', 'temperature', 'enable_thinking')
                if key in source})
        requested = safe.get('enable_thinking')
        actual = _effective_thinking(self, requested)
        if type(actual) is bool:
            safe['enable_thinking'] = actual
            safe['thinking_mode'] = _thinking_mode(actual)
            if getattr(getattr(self, 'settings', None), 'provider', None) == 'vllm':
                safe['extra_body'] = {
                    'chat_template_kwargs': {'enable_thinking': actual}
                }
        encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True)
        identity = {'request_id': uuid.uuid4().hex, 'stage': stage,
                    'path_name': safe.get('name'),
                    'prompt_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
                    'enable_thinking': actual,
                    'thinking_mode': _thinking_mode(actual)}
        token = _request.set(identity)
        started = time.monotonic()
        try:
            try:
                root = Path(os.environ['QWEN_DIAGNOSTICS_DIR'])
                root.mkdir(parents=True, exist_ok=True)
                (root / (identity['request_id'] + '.json')).write_text(encoded, encoding='utf-8')
            except OSError as exc:
                _report_sink_failure(exc)
            prompt_chars = sum(
                len(message.get('content', ''))
                for message in safe.get('messages', [])
                if isinstance(message, dict) and isinstance(message.get('content', ''), str)
            )
            safe_emit('request_started', queue_seconds=_queue.get(), prompt_chars=prompt_chars)
            result = fn(self, stage, payload)
            result_metadata = getattr(result, 'provider_metadata', {})
            if not isinstance(result_metadata, dict):
                result_metadata = {}
            safe_emit('request_completed', elapsed_seconds=time.monotonic() - started,
                 usage=result_metadata.get('usage'),
                 enable_thinking=result_metadata.get('enable_thinking', actual),
                 thinking_mode=result_metadata.get(
                     'thinking_mode', _thinking_mode(actual)))
            return result
        except BaseException as exc:
            safe_emit('request_failed', elapsed_seconds=time.monotonic() - started,
                 exception_chain=exception_chain(exc), usage=None,
                 enable_thinking=actual, thinking_mode=_thinking_mode(actual))
            raise
        finally:
            _request.reset(token)
    return invoke


def diagnostic_http_client():
    from openai import DefaultHttpxClient
    import httpx

    class _DiagnosticSyncByteStream(httpx.SyncByteStream):
        def __init__(self, stream, root, request_identity, sentence_identity):
            self._stream = stream
            self._root = root
            self._request_identity = dict(request_identity)
            self._sentence_identity = dict(sentence_identity)
            self._path = root / (request_identity['request_id'] + '.stream.bin')
            self._chunk_index = 0

        def _emit(self, event, **fields):
            request_token = _request.set(self._request_identity)
            sentence_token = _sentence.set(self._sentence_identity)
            try:
                safe_emit(event, **fields)
            finally:
                _sentence.reset(sentence_token)
                _request.reset(request_token)

        def _append(self, chunk):
            try:
                self._root.mkdir(parents=True, exist_ok=True)
                with _lock:
                    with self._path.open('ab') as handle:
                        handle.write(chunk)
                        handle.flush()
            except Exception as exc:
                _report_sink_failure(exc)

        def _streamed_bytes(self):
            try:
                return self._path.stat().st_size
            except OSError:
                return 0

        def __iter__(self):
            try:
                for chunk in self._stream:
                    data = bytes(chunk)
                    self._append(data)
                    self._emit('http_stream_chunk', chunk_index=self._chunk_index,
                               chunk_bytes=len(data))
                    self._chunk_index += 1
                    yield chunk
            except BaseException as exc:
                self._emit('http_stream_failed', chunk_count=self._chunk_index,
                           streamed_bytes=self._streamed_bytes(),
                           exception_chain=exception_chain(exc))
                raise

        def close(self):
            close = getattr(self._stream, 'close', None)
            if callable(close):
                close()
            self._emit('http_stream_closed', chunk_count=self._chunk_index,
                       streamed_bytes=self._streamed_bytes())

    class Client(DefaultHttpxClient):
        def send(self, request, *args, **kwargs):
            started = time.monotonic()
            retry = request.headers.get('x-stainless-retry-count', '0')
            safe_emit('http_started', retry_index=int(retry))
            try:
                response = super().send(request, *args, **kwargs)
                usage, output_lengths = None, None
                if response.status_code == 200 and not kwargs.get('stream', False):
                    try:
                        body = response.json()
                        usage = body.get('usage')
                        message = body.get('choices', [{}])[0].get('message', {})
                        output_lengths = {k: len(v) for k, v in message.items()
                                          if isinstance(v, str)}
                    except Exception:
                        # Optional evidence must not convert a received response
                        # into an SDK transport retry; adapter validation owns it.
                        pass
                if kwargs.get('stream', False) and os.environ.get('QWEN_DIAGNOSTICS_DIR'):
                    request_identity = dict(_request.get())
                    request_id = request_identity.get('request_id')
                    if request_id:
                        response.stream = _DiagnosticSyncByteStream(
                            response.stream,
                            Path(os.environ['QWEN_DIAGNOSTICS_DIR']),
                            request_identity,
                            dict(_sentence.get()),
                        )
                safe_emit('http_completed', retry_index=int(retry),
                     elapsed_seconds=time.monotonic() - started,
                     status_code=response.status_code, usage=usage,
                     output_lengths=output_lengths)
                return response
            except BaseException as exc:
                safe_emit('http_failed', retry_index=int(retry),
                     elapsed_seconds=time.monotonic() - started,
                     exception_chain=exception_chain(exc), usage=None)
                raise
    return Client(trust_env=False)
