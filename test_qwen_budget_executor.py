import math
import subprocess
import time

import pytest

import qwen_budget_executor as executor
from qwen_budget_executor import remaining_seconds, run_bounded


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(time, 'monotonic', lambda: 0)


def test_remaining_budget_includes_reserve():
    assert remaining_seconds(110, 10, 5, 1.25) == 57600


def test_shared_ledger_reuses_one_absolute_deadline_and_rejects_budget_reset(tmp_path):
    ledger_path = tmp_path / 'budget-ledger.json'
    ledger, initial_seconds = executor._open_budget_ledger(
        ledger_path, total=110, spent=20, hourly_rate=5.59,
        reserve_factor=1.25, billing_basis='conservative-estimate',
    )
    assert ledger['max_seconds'] == initial_seconds

    # Simulate time already consumed by an earlier diagnosis phase. A later
    # phase must use the persisted deadline, not recompute a fresh 90-yuan run.
    ledger['deadline_at'] = (
        executor.datetime.now(executor.timezone.utc)
        + executor.timedelta(seconds=3)
    ).isoformat()
    executor._atomic_json_write(ledger_path, ledger)
    resumed, seconds = executor._open_budget_ledger(
        ledger_path, total=110, spent=20.01, hourly_rate=5.59,
        reserve_factor=1.25, billing_basis='conservative-estimate',
    )
    assert resumed['reported_spent_yuan'] == 20.01
    assert 1 <= seconds <= 3

    with pytest.raises(ValueError, match='lower cumulative spent'):
        executor._open_budget_ledger(
            ledger_path, total=110, spent=19.99, hourly_rate=5.59,
            reserve_factor=1.25, billing_basis='conservative-estimate',
        )


@pytest.mark.parametrize('spent', [math.nan, math.inf, -1, 110])
def test_existing_ledger_rejects_invalid_current_spend(tmp_path, spent):
    ledger_path = tmp_path / 'budget-ledger.json'
    executor._open_budget_ledger(
        ledger_path, total=110, spent=20, hourly_rate=5.59,
        reserve_factor=1.25, billing_basis='conservative-estimate',
    )
    with pytest.raises(ValueError):
        executor._open_budget_ledger(
            ledger_path, total=110, spent=spent, hourly_rate=5.59,
            reserve_factor=1.25, billing_basis='conservative-estimate',
        )


def test_ledger_write_failure_cannot_mask_runner_error(tmp_path, monkeypatch):
    original = OSError('runner failure')
    calls = []

    def fail_runner(argv):
        raise original

    def fail_failed_phase(path, ledger, *, status, **kwargs):
        calls.append(status)
        if status == 'failed':
            raise PermissionError('ledger failure')

    monkeypatch.setattr(subprocess, 'Popen', fail_runner)
    monkeypatch.setattr(executor, '_record_ledger_phase', fail_failed_phase)
    with pytest.raises(OSError) as caught:
        executor.main([
            '--total-yuan', '110', '--spent-yuan', '10',
            '--hourly-rate-yuan', '5', '--billing-basis',
            'conservative-estimate', '--report', str(tmp_path / 'report.json'),
            '--ledger', str(tmp_path / 'ledger.json'), '--', '--phase', 'diagnosis',
        ])
    assert caught.value is original
    assert calls == ['running', 'failed']


@pytest.mark.parametrize('values', [
    (110, 110, 5, 1.25), (110, 111, 5, 1.25), (110, -1, 5, 1.25),
    (110, 0, 0, 1.25), (110, 0, 5, 0.9), (math.inf, 0, 5, 1.25),
    (110, math.nan, 5, 1.25)])
def test_invalid_or_exhausted_budget_fails_closed(values):
    with pytest.raises(ValueError):
        remaining_seconds(*values)


def test_deadline_kills_and_reaps_direct_runner(monkeypatch):
    class Child:
        def __init__(self):
            self.killed = False
            self.waits = []
        def wait(self, timeout=None):
            self.waits.append(timeout)
            if not self.killed:
                raise subprocess.TimeoutExpired('runner', timeout)
            assert self.killed
            return 1
        def kill(self):
            self.killed = True
    child = Child()
    monkeypatch.setattr(subprocess, 'Popen', lambda argv: child)
    assert run_bounded(['python', 'run_multiseed.py'], 10) == 124
    assert child.killed and child.waits == [10, 5]


def test_success_does_not_kill_runner(monkeypatch):
    class Child:
        def wait(self, timeout=None):
            return 0
        def kill(self):
            pytest.fail('successful child must not be killed')
    monkeypatch.setattr(subprocess, 'Popen', lambda argv: Child())
    assert run_bounded(['python', 'run_multiseed.py'], 10) == 0


def test_expired_deadline_never_launches(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda argv: pytest.fail('must not launch'))
    assert run_bounded(['runner'], 10, deadline=-1) == 124


def test_slow_launch_is_immediately_stopped(monkeypatch):
    class Child:
        killed = False
        def kill(self):
            self.killed = True
        def wait(self, timeout=None):
            assert self.killed
            return 1
    child = Child()
    def launch(argv):
        monkeypatch.setattr(time, 'monotonic', lambda: 20)
        return child
    monkeypatch.setattr(subprocess, 'Popen', launch)
    assert run_bounded(['runner'], 10) == 124
    assert child.killed


def test_interrupt_is_preserved_when_cleanup_fails(monkeypatch):
    original = KeyboardInterrupt()
    class Child:
        def wait(self, timeout=None):
            raise original
        def kill(self):
            raise OSError('kill failed')
    monkeypatch.setattr(subprocess, 'Popen', lambda argv: Child())
    with pytest.raises(KeyboardInterrupt) as caught:
        run_bounded(['runner'], 10)
    assert caught.value is original


def test_launch_failure_records_terminal_status(tmp_path, monkeypatch):
    import json
    report = tmp_path / 'budget.json'
    def fail(argv):
        raise OSError('launch failed')
    monkeypatch.setattr(subprocess, 'Popen', fail)
    with pytest.raises(OSError):
        executor.main(['--total-yuan','110','--spent-yuan','10','--hourly-rate-yuan','5',
                       '--billing-basis','conservative-estimate','--report',str(report),
                       '--','--phase','provider-cache'])
    saved = json.loads(report.read_text())
    assert saved['status'] == 'failed' and saved['exception_type'] == 'OSError'


def test_report_delay_can_exhaust_budget_before_launch(tmp_path, monkeypatch):
    import json
    report = tmp_path / 'budget.json'
    def delayed_print(*args, **kwargs):
        monkeypatch.setattr(time, 'monotonic', lambda: 20)
    monkeypatch.setattr(executor, 'print', delayed_print, raising=False)
    monkeypatch.setattr(subprocess, 'Popen', lambda argv: pytest.fail('expired before launch'))
    code = executor.main(['--total-yuan','110','--spent-yuan','100','--hourly-rate-yuan','3600',
                          '--reserve-factor','1','--billing-basis','conservative-estimate',
                          '--report',str(report),'--','--phase','provider-cache'])
    assert code == 124
    assert json.loads(report.read_text())['status'] == 'deadline'


def test_report_failure_cannot_mask_launch_error(tmp_path, monkeypatch):
    from pathlib import Path
    original = OSError('launch failure')
    def fail_launch(argv):
        raise original
    def fail_report(*args, **kwargs):
        raise PermissionError('report failure')
    monkeypatch.setattr(subprocess, 'Popen', fail_launch)
    monkeypatch.setattr(Path, 'write_text', fail_report)
    with pytest.raises(OSError) as caught:
        executor.main(['--total-yuan','110','--spent-yuan','10','--hourly-rate-yuan','5',
                       '--billing-basis','conservative-estimate','--report',str(tmp_path / 'budget.json'),
                       '--','--phase','provider-cache'])
    assert caught.value is original
