"""Bound one direct experiment runner by explicit remaining rental budget.

This stops client requests, not cloud billing. Supply cumulative spend through
launch time, including earlier hosts, diagnostics, idle time, and reruns.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid


LEDGER_SCHEMA = 'qwen-budget-ledger-v1'


def remaining_seconds(total, spent, hourly_rate, reserve_factor=1.25):
    if (not all(math.isfinite(v) for v in (total, spent, hourly_rate, reserve_factor))
            or total <= 0 or spent < 0 or spent >= total or hourly_rate <= 0
            or reserve_factor < 1):
        raise ValueError('invalid or exhausted rental budget')
    seconds = math.floor((total - spent) / (hourly_rate * reserve_factor) * 3600)
    if seconds < 1:
        raise ValueError('remaining rental budget is less than one second')
    return seconds


def _stop(child):
    try:
        child.kill()
    except Exception:
        pass  # It may already have exited; still attempt a bounded reap.
    child.wait(timeout=5)


def run_bounded(command, seconds, *, deadline=None):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('hard deadline must be positive and finite')
    deadline = time.monotonic() + seconds if deadline is None else deadline
    if deadline <= time.monotonic():
        return 124
    # Invoke Python directly, never a shell or background launcher. Killing this
    # process closes provider sockets; incomplete cells remain unpublished.
    child = subprocess.Popen(command)
    try:
        left = deadline - time.monotonic()
        if left <= 0:
            _stop(child)
            return 124
        return child.wait(timeout=left)
    except subprocess.TimeoutExpired:
        _stop(child)
        return 124
    except BaseException:
        try:
            _stop(child)
        except BaseException:
            pass  # Preserve the original interruption, not a cleanup failure.
        raise


def _atomic_json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ledger_deadline_seconds(ledger):
    try:
        deadline = datetime.fromisoformat(ledger['deadline_at'])
        seconds = math.floor((deadline - datetime.now(timezone.utc)).total_seconds())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('budget ledger has an invalid absolute deadline') from exc
    if seconds < 1:
        raise ValueError('shared budget ledger deadline is exhausted')
    return seconds


def _previous_ledger_link(path, *, total, hourly_rate, reserve_factor,
                          billing_basis):
    previous_path = Path(path)
    try:
        content = previous_path.read_bytes()
        previous = json.loads(content.decode('utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('previous budget ledger is unreadable') from exc
    if not isinstance(previous, dict) or previous.get('schema') != LEDGER_SCHEMA:
        raise ValueError('previous budget ledger schema is invalid')
    expected = {
        'total_yuan': total,
        'hourly_rate_yuan': hourly_rate,
        'reserve_factor': reserve_factor,
        'billing_basis': billing_basis,
    }
    if any(previous.get(key) != value for key, value in expected.items()):
        raise ValueError('previous budget ledger does not match this continuation')
    previous_spent = previous.get('reported_spent_yuan')
    if (type(previous_spent) not in (int, float)
            or not math.isfinite(previous_spent) or previous_spent < 0
            or previous_spent >= total):
        raise ValueError('previous budget ledger spend is invalid')
    return {
        'previous_ledger_sha256': hashlib.sha256(content).hexdigest(),
        'previous_reported_spent_yuan': float(previous_spent),
    }


def _open_budget_ledger(path, *, total, spent, hourly_rate, reserve_factor,
                        billing_basis, previous_ledger=None):
    """Create or resume one cumulative budget deadline across all phases."""
    if (not all(math.isfinite(value) for value in
                 (total, spent, hourly_rate, reserve_factor))
            or total <= 0 or spent < 0 or spent >= total or hourly_rate <= 0
            or reserve_factor < 1):
        raise ValueError('invalid or exhausted rental budget')
    path = Path(path)
    continuation = None
    if previous_ledger is not None:
        previous_path = Path(previous_ledger)
        if previous_path.resolve() == path.resolve():
            raise ValueError('continuation ledger must not overwrite its predecessor')
        continuation = _previous_ledger_link(
            previous_path, total=total, hourly_rate=hourly_rate,
            reserve_factor=reserve_factor, billing_basis=billing_basis,
        )
        if spent < continuation['previous_reported_spent_yuan']:
            raise ValueError(
                'budget ledger rejects a lower cumulative spent value than its predecessor'
            )
    if path.exists():
        try:
            ledger = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError('budget ledger is unreadable') from exc
        if not isinstance(ledger, dict) or ledger.get('schema') != LEDGER_SCHEMA:
            raise ValueError('budget ledger schema is invalid')
        recorded_continuation = ledger.get('continuation')
        if continuation is None and recorded_continuation is not None:
            raise ValueError('continuation ledger requires its predecessor')
        if continuation is not None and recorded_continuation != continuation:
            raise ValueError('continuation ledger predecessor does not match')
        expected = {
            'total_yuan': total,
            'hourly_rate_yuan': hourly_rate,
            'reserve_factor': reserve_factor,
            'billing_basis': billing_basis,
        }
        for key, value in expected.items():
            if ledger.get(key) != value:
                raise ValueError(f'budget ledger {key} does not match this launch')
        recorded_spent = ledger.get('reported_spent_yuan')
        if (type(recorded_spent) not in (int, float)
                or not math.isfinite(recorded_spent)
                or spent < recorded_spent):
            raise ValueError(
                'budget ledger rejects a lower cumulative spent value; phases share one ledger'
            )
        ledger['reported_spent_yuan'] = max(float(spent), float(recorded_spent))
        ledger.setdefault('phases', [])
        if not isinstance(ledger['phases'], list):
            raise ValueError('budget ledger phases must be a list')
        return ledger, _ledger_deadline_seconds(ledger)

    seconds = remaining_seconds(total, spent, hourly_rate, reserve_factor)
    started = datetime.now(timezone.utc)
    ledger = {
        'schema': LEDGER_SCHEMA,
        'total_yuan': total,
        'initial_spent_yuan': spent,
        'reported_spent_yuan': spent,
        'hourly_rate_yuan': hourly_rate,
        'reserve_factor': reserve_factor,
        'billing_basis': billing_basis,
        'max_seconds': seconds,
        'started_at': started.isoformat(),
        'deadline_at': (started + timedelta(seconds=seconds)).isoformat(),
        'phases': [],
    }
    if continuation is not None:
        ledger['continuation'] = continuation
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(ledger, indent=2))
    except FileExistsError:
        return _open_budget_ledger(
            path, total=total, spent=spent, hourly_rate=hourly_rate,
            reserve_factor=reserve_factor, billing_basis=billing_basis,
            previous_ledger=previous_ledger,
        )
    return ledger, seconds


def _record_ledger_phase(path, ledger, *, status, exit_code=None,
                         exception_type=None, report=None):
    phase = {
        'phase_id': uuid.uuid4().hex,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'status': status,
    }
    if exit_code is not None:
        phase['exit_code'] = exit_code
    if exception_type is not None:
        phase['exception_type'] = exception_type
    if report is not None:
        phase['report'] = str(report)
    ledger.setdefault('phases', []).append(phase)
    _atomic_json_write(path, ledger)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-yuan', type=float, required=True)
    parser.add_argument('--spent-yuan', type=float, required=True)
    parser.add_argument('--hourly-rate-yuan', type=float, required=True)
    parser.add_argument('--reserve-factor', type=float, default=1.25)
    parser.add_argument('--billing-basis', choices=['verified', 'conservative-estimate'], required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, default=None,
                        help='Shared cumulative ledger for diagnosis, smoke, and reruns.')
    parser.add_argument('--previous-ledger', type=Path, default=None,
                        help='Immutable predecessor ledger for a budget continuation.')
    parser.add_argument('--runner-script', type=Path, default=None,
                        help='Python runner to execute; defaults to run_multiseed.py.')
    parser.add_argument('runner_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    ledger = None
    ledger_seconds = None
    if args.previous_ledger is not None and args.ledger is None:
        parser.error('--previous-ledger requires --ledger')
    if args.ledger is not None:
        ledger, ledger_seconds = _open_budget_ledger(
            args.ledger, total=args.total_yuan, spent=args.spent_yuan,
            hourly_rate=args.hourly_rate_yuan, reserve_factor=args.reserve_factor,
            billing_basis=args.billing_basis,
            previous_ledger=args.previous_ledger,
        )
    seconds = ledger_seconds if ledger_seconds is not None else remaining_seconds(
        args.total_yuan, args.spent_yuan, args.hourly_rate_yuan, args.reserve_factor
    )
    runner_args = args.runner_args[1:] if args.runner_args[:1] == ['--'] else args.runner_args
    if not runner_args:
        parser.error('provide run_multiseed.py arguments after --')
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + seconds
    report = {
        'total_yuan': args.total_yuan, 'spent_yuan': args.spent_yuan,
        'hourly_rate_yuan': args.hourly_rate_yuan, 'reserve_factor': args.reserve_factor,
        'billing_basis': args.billing_basis, 'max_seconds': seconds,
        'started_at': started.isoformat(),
        'hard_deadline': (started + timedelta(seconds=seconds)).isoformat(),
        'status': 'running', 'stops_instance_billing': False,
    }
    if args.ledger is not None:
        report['ledger'] = str(args.ledger)
        report['shared_deadline'] = ledger['deadline_at']
    runner_script = (args.runner_script if args.runner_script is not None
                     else Path(__file__).with_name('run_multiseed.py'))
    report['runner_script'] = str(runner_script)
    # Exclusive create: never overwrite evidence from another launch.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2))
    if ledger is not None:
        _record_ledger_phase(args.ledger, ledger, status='running', report=args.report)
    original_error = None
    try:
        print(json.dumps(report), flush=True)
        code = run_bounded(
            [sys.executable, str(runner_script), *runner_args],
            seconds, deadline=deadline,
        )
        report.update(exit_code=code, status='deadline' if code == 124 else 'finished')
    except BaseException as exc:
        original_error = exc
        report.update(status='failed', exception_type=type(exc).__name__)
        if ledger is not None:
            try:
                _record_ledger_phase(
                    args.ledger, ledger, status='failed',
                    exception_type=type(exc).__name__, report=args.report,
                )
            except Exception:
                # Evidence persistence must never replace the first provider
                # or runner exception that caused this launch to fail.
                pass
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        try:
            args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        except Exception:
            if original_error is None:
                raise
        if ledger is not None and original_error is None:
            _record_ledger_phase(
                args.ledger, ledger, status=report['status'],
                exit_code=report.get('exit_code'), report=args.report,
            )
    return code


if __name__ == '__main__':
    raise SystemExit(main())
