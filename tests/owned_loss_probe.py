"""Standalone observer: loss of an actual provider supervisor never grants reuse."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from kilix_qwen_tts.owned import OwnedExecution
from kilix_qwen_tts.protocol import ProtocolError, receive_packet
from kilix_qwen_tts.service import client_request, request_value
from test_leased_runtime import prepared, raw, waiting, until
from voicelib import device_leases


def children(pid):
    found = set()
    for task in Path(f'/proc/{pid}/task').glob('*'):
        try:
            found.update(int(value) for value in (task/'children').read_text().split())
        except FileNotFoundError:
            pass
    return found


def tree(pid):
    found, pending = set(), [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        pending.extend(children(current))
    return found


def reap_owned(unrelated):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        owned = children(os.getpid())-{unrelated}
        for pid in owned:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        if not owned:
            return
        time.sleep(.005)
    raise AssertionError('observer owned tree survived explicit cleanup')


def main():
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    records = []
    original = OwnedExecution.spawn
    def spawn(owner, *args, **kwargs):
        identity = os.fstat(owner.lease.guard_fd)
        process = original(owner, *args, **kwargs)
        records.append((process, (identity.st_dev, identity.st_ino)))
        return process
    result = {}
    try:
        with prepared('escape') as fixture:
            try:
                with patch.object(OwnedExecution, 'spawn', spawn), raw(fixture, 'escape') as channel:
                    until(lambda: waiting(fixture))
                    assert len(records) == 1
                    supervisor, identity = records[0]
                    owned = tree(supervisor.pid)
                    holders = []
                    for pid in owned:
                        for fd in Path(f'/proc/{pid}/fd').glob('*'):
                            try:
                                info = fd.stat()
                            except FileNotFoundError:
                                continue
                            if (info.st_dev, info.st_ino) == identity:
                                holders.append(pid)
                    assert supervisor.pid in holders and len(set(holders)) >= 2
                    os.kill(supervisor.pid, signal.SIGKILL)
                    event, descriptors = receive_packet(channel)
                    assert not descriptors and event['type'] == 'error'
                    assert event['error']['code'] == 'SUPERVISOR_FAILED', event
                    status = client_request(fixture.ipc, request_value('status'))
                    assert status['provider_state'] == 'unavailable', status
                    for operation in ('unload',):
                        try:
                            client_request(fixture.ipc, request_value(operation))
                        except ProtocolError as error:
                            assert error.code == 'SUPERVISOR_FAILED'
                        else:
                            raise AssertionError('unproven service accepted unload')
                    survivors = [pid for pid in owned if Path(f'/proc/{pid}').exists()]
                    survivor_states = []
                    for pid in survivors:
                        try:
                            state = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[0]
                        except FileNotFoundError:
                            continue
                        inherited = False
                        for fd in Path(f'/proc/{pid}/fd').glob('*'):
                            try:
                                info = fd.stat()
                            except FileNotFoundError:
                                continue
                            inherited |= (info.st_dev, info.st_ino) == identity
                        survivor_states.append({'state': state, 'guard': inherited})
                    assert unrelated.poll() is None
                    result.update(guard_holders=len(set(holders)), owned_before_loss=len(owned),
                                  terminal=event['error']['code'], status=status['provider_state'],
                                  observed_survivors_at_terminal=len(survivors),
                                  survivor_states_at_terminal=survivor_states)
                reap_owned(unrelated.pid)
                assert unrelated.poll() is None
                assert children(os.getpid()) == {unrelated.pid}
                try:
                    lease = device_leases.acquire(job_id='after-all-fds-closed', workload='llm-turn',
                        device='other-alias', deadline=time.monotonic()+.15,
                        namespace=str(fixture.root/'shared-lease'))
                except device_leases.LeaseError as error:
                    assert error.code == 'unavailable', error.code
                    result['successor_after_complete_observer_cleanup'] = error.code
                else:
                    lease.release(cleanup_complete=True)
                    raise AssertionError('unacknowledged grant was handed off')
                result['unrelated_preserved'] = unrelated.poll() is None
            finally:
                reap_owned(unrelated.pid)
    finally:
        if unrelated.poll() is None:
            unrelated.kill()
        unrelated.wait(timeout=5)
        reap_owned(-1)
        result['children_after_probe_cleanup'] = sorted(children(os.getpid()))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
