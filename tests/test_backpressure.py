"""Real saturated sockets must leave owned cleanup reachable without a reader."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import select
import socket
import subprocess
import time
import unittest
from unittest.mock import patch

import kilix_qwen_tts.service as service_module
from kilix_qwen_tts.protocol import receive_packet, send_packet
from kilix_qwen_tts.runtime import tree_digest
from kilix_qwen_tts.service import SOCKET_NAME, client_request, request_value
from kilix_qwen_tts.streaming import ClientStream
import test_runtime as fixtures


TAIL = '''
import struct
signal.signal(signal.SIGTERM,signal.SIG_IGN)
child=subprocess.Popen(['/usr/bin/python3','-c',
    'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],
    start_new_session=True)
pcm=b'\\x01\\x00'*24
payload=payload[:44]+pcm*512
payload=payload[:4]+struct.pack('<I',len(payload)-8)+payload[8:40]+struct.pack('<I',len(payload)-44)+payload[44:]
destination.write_bytes(payload)
result['duration_ms']=512
result['audio']={'byte_length':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
out=sys.stdout.buffer
out.write(b'KILIX_PCM_V1\\x00')
for i in range(512):out.write(struct.pack('!cI',b'P',len(pcm))+pcm)
out.flush()
pathlib.Path('/job/saturated').touch()
while not pathlib.Path('/job/continue').exists():time.sleep(.005)
final=json.dumps(result).encode()
out.write(struct.pack('!cI',b'R',len(final))+final);out.flush()
'''


def descendants(pid):
    found = set()
    for path in Path(f'/proc/{pid}/task').glob('*/children'):
        try:
            found.update(int(value) for value in path.read_text().split())
        except FileNotFoundError:
            pass
    return found | {child for direct in tuple(found) for child in descendants(direct)}


def wait_until(predicate, seconds=3):
    until = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= until:
            raise AssertionError('bounded observation did not complete')
        time.sleep(.005)


@contextmanager
def saturated(timeout=10):
    before_fds = len(os.listdir('/proc/self/fd'))
    before_children = descendants(os.getpid())
    fixture = fixtures.RuntimeTests('runTest')
    fixture.setUp()
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sender = []
    original = service_module.send_packet

    def observe(channel, value, descriptor=None):
        if value['type'] == 'accepted':
            channel.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            sender.append(channel)
        return original(channel, value, descriptor)

    try:
        engine = fixture.runtime.python_root / 'bin/python3.12'
        engine.write_text(engine.read_text().replace('print(json.dumps(result))', TAIL))
        fixture.manifest['environment']['python_sha256'] = hashlib.sha256(engine.read_bytes()).hexdigest()
        fixture.manifest['environment']['python_root_sha256'] = tree_digest(fixture.runtime.python_root)
        with patch.object(service_module, 'send_packet', side_effect=observe):
            fixture.start()
            peer.settimeout(3)
            peer.connect(str(fixture.ipc / SOCKET_NAME))
            request = request_value('submit', job_id='saturated', args=fixture.arguments(), timeout=timeout, stream=True)
            started = time.monotonic()
            with fixture.audio.open('rb') as audio:
                send_packet(peer, request, audio.fileno())
            accepted, fds = receive_packet(peer)
            assert accepted['type'] == 'accepted' and not fds
            wait_until(lambda: bool(list(fixture.jobs.glob('*/saturated')))
                       and bool(sender) and not select.select([], sender, [], 0)[1])
            owned = descendants(os.getpid()) - before_children
            assert len(owned) >= 4, owned
            yield fixture, peer, owned, started
    finally:
        # An assertion failure still closes the original client before fixture
        # teardown; passing cancel/stop checks occur while it remains unread.
        peer.close()
        fixture.doCleanups()
        assert descendants(os.getpid()) == before_children
        assert len(os.listdir('/proc/self/fd')) == before_fds


class BackpressureTests(unittest.TestCase):
    def assert_reaped(self, fixture, owned):
        wait_until(lambda: not fixture.service._jobs, 2)
        self.assertEqual(list(fixture.jobs.iterdir()), [])
        self.assertTrue(all(not Path(f'/proc/{pid}').exists() for pid in owned))

    def test_cancel_reaps_escaped_tree_while_peer_stays_unread(self):
        unrelated = subprocess.Popen(['/usr/bin/python3', '-c', 'import time;time.sleep(60)'])
        try:
            with saturated() as (fixture, peer, owned, _started):
                reply = client_request(fixture.ipc, request_value('cancel', job_id='saturated'))
                self.assertTrue(reply['cancel_requested'])
                self.assert_reaped(fixture, owned)
                self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'ready')
                self.assertIsNone(unrelated.poll())
                self.assertGreaterEqual(peer.fileno(), 0)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_shutdown_finishes_while_peer_stays_unread(self):
        with saturated() as (fixture, peer, owned, _started):
            fixture.service.stop()
            self.assert_reaped(fixture, owned)
            fixture.thread.join(2)
            self.assertFalse(fixture.thread.is_alive())
            self.assertGreaterEqual(peer.fileno(), 0)

    def test_disconnect_and_absolute_deadline_reap_saturated_tree(self):
        for mode in ('disconnect', 'deadline'):
            with self.subTest(mode=mode), saturated(timeout=1 if mode == 'deadline' else 10) as (fixture, peer, owned, started):
                if mode == 'disconnect':
                    peer.close()
                self.assert_reaped(fixture, owned)
                self.assertLess(time.monotonic() - started, 2)

    def test_resumed_reader_receives_each_descriptor_once_and_exact_final_pcm(self):
        with saturated() as (fixture, peer, owned, _started):
            # At least two real socket timeouts must elapse before resuming.
            time.sleep(.15)
            (next(fixture.jobs.iterdir()) / 'continue').touch()
            stream = ClientStream(1000)
            chunks = 0
            while True:
                event, fds = receive_packet(peer)
                try:
                    if event['type'] == 'chunk':
                        stream.append(event['result'], fds)
                        chunks += 1
                    else:
                        self.assertEqual(event['type'], 'result')
                        self.assertEqual(len(fds), 1)
                        audio = os.pread(fds[0], event['result']['audio']['byte_length'] + 1, 0)
                        stream.require_audio(audio)
                        self.assertEqual(event['result']['audio']['sha256'], hashlib.sha256(audio).hexdigest())
                        self.assert_reaped(fixture, owned)
                        break
                finally:
                    for fd in fds:
                        os.close(fd)
            self.assertEqual(chunks, 512)


if __name__ == '__main__':
    unittest.main()
