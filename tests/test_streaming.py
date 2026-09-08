"""Partial-write, descriptor, final-binding and real owned-process controls."""
from contextlib import contextmanager
import copy
import hashlib
import io
import os
import tempfile
import time
import unittest

from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.runtime import tree_digest
from kilix_qwen_tts.service import client_request, request_value
from kilix_qwen_tts.streaming import (MAGIC, HEADER, ClientStream, WorkerStream,
                                    WorkerStreamReader, chunk_metadata, requested, wave_from_pcm)
import test_runtime as fixtures


class RecordTests(unittest.TestCase):
    def test_bytewise_partial_records_emit_only_complete_pcm(self):
        encoded = io.BytesIO()
        writer = WorkerStream(encoded)
        pcm = b"\x01\x00" * 24
        writer.pcm(pcm)
        writer.pcm(pcm)
        writer.finish({"complete": True})
        seen = []
        with tempfile.TemporaryFile() as source:
            reader = WorkerStreamReader(source.fileno(), 10, lambda *row: seen.append(row), lambda: None)
            for byte in encoded.getvalue():
                source.write(bytes([byte]))
                source.flush()
                reader.drain()
            reader.drain(final=True)
            self.assertEqual(seen, [(0, 0, pcm), (1, 24, pcm)])
            self.assertEqual(reader.result, {"complete": True})
            reader.require_audio(wave_from_pcm(pcm * 2))
            with self.assertRaises(ProtocolError):
                reader.require_audio(wave_from_pcm(b"\x00" * 96))

    def test_truncated_oversize_extension_and_trailing_records_refuse(self):
        encoded = io.BytesIO()
        writer = WorkerStream(encoded)
        writer.pcm(b"\0" * 48)
        writer.finish({"done": True})
        populations = [b"", MAGIC[:-1], MAGIC + HEADER.pack(b"P", 0),
                       MAGIC + HEADER.pack(b"P", 0xffffffff), MAGIC + HEADER.pack(b"P", 47),
                       MAGIC + HEADER.pack(b"X", 1) + b"x", encoded.getvalue()[:-1],
                       encoded.getvalue() + b"\0", encoded.getvalue() + HEADER.pack(b"P", 48) + b"\0" * 48]
        for result in (b'{"x":NaN}', b'{"x":1,"x":2}', b'[]'):
            populations.append(MAGIC + HEADER.pack(b"P", 48) + b"\0" * 48 + HEADER.pack(b"R", len(result)) + result)
        for value in populations:
            with self.subTest(bytes=len(value)), tempfile.TemporaryFile() as source:
                source.write(value)
                source.flush()
                reader = WorkerStreamReader(source.fileno(), 2, lambda *_: None, lambda: None)
                with self.assertRaises(ProtocolError):
                    reader.drain(final=True)

    def test_duration_shrink_and_cancel_are_bounded(self):
        with tempfile.TemporaryFile() as source:
            writer = WorkerStream(source)
            writer.pcm(b"\0" * 96)
            reader = WorkerStreamReader(source.fileno(), 1, lambda *_: None, lambda: None)
            with self.assertRaises(ProtocolError):
                reader.drain()
            reader = WorkerStreamReader(source.fileno(), 10, lambda *_: None, lambda: None)
            reader.drain()
            source.truncate(0)
            with self.assertRaises(ProtocolError):
                reader.drain()
            def cancelled():
                raise ProtocolError("CANCELED", "cancelled control")
            reader = WorkerStreamReader(source.fileno(), 10, lambda *_: None, cancelled)
            with self.assertRaises(ProtocolError) as error:
                reader.drain()
            self.assertEqual(error.exception.code, "CANCELED")

    def test_chunk_metadata_fd_bytes_and_final_audio_are_bound(self):
        pcm = b"\x01\0" * 24
        metadata = chunk_metadata(0, 0, pcm)
        with tempfile.TemporaryFile() as source:
            source.write(pcm)
            source.flush()
            fd = os.open(f"/proc/self/fd/{source.fileno()}", os.O_RDONLY | os.O_CLOEXEC)
            try:
                reader = ClientStream(1)
                reader.append(metadata, (fd,))
                reader.require_audio(wave_from_pcm(pcm))
                with self.assertRaises(ProtocolError):
                    reader.require_audio(wave_from_pcm(b"\0" * 48))
                for key, value in (("sequence", 1), ("frame_offset", 1), ("frame_count", True),
                                   ("channels", 2), ("sample_format", []), ("sample_rate_hz", 48_000)):
                    changed = copy.deepcopy(metadata)
                    changed[key] = value
                    with self.subTest(key=key), self.assertRaises(ProtocolError):
                        ClientStream(1).append(changed, (fd,))
                with self.assertRaises(ProtocolError):
                    ClientStream(1).append(metadata, (source.fileno(),))
                os.pwrite(source.fileno(), b"\0" * 48, 0)
                with self.assertRaises(ProtocolError):
                    ClientStream(1).append(metadata, (fd,))
            finally:
                os.close(fd)

    def test_extension_is_explicit_and_typed(self):
        self.assertFalse(requested({"op": "submit", "extensions": {"x_future": True}}))
        self.assertTrue(requested({"op": "submit", "extensions": {"x_pcm_stream_v1": True}}))
        for value in (1, None, [], {}, "true"):
            with self.assertRaises(ProtocolError):
                requested({"op": "submit", "extensions": {"x_pcm_stream_v1": value}})
        with self.assertRaises(ProtocolError):
            requested({"op": "status", "extensions": {"x_pcm_stream_v1": True}})


STREAM_TAIL = '''
if request.get('streaming_pcm'):
    import struct
    output=sys.stdout.buffer
    output.write(b'KILIX_PCM_V1\\x00')
    pcm=payload[44:]
    for offset in (0,2400):
        part=pcm[offset:offset+2400]
        output.write(struct.pack('!cI',b'P',len(part))+part);output.flush()
        if offset==0:
            if args['text']=='stream-hold':time.sleep(60)
            if args['text']=='stream-truncated':raise SystemExit(0)
            time.sleep(.15)
    if args['text']=='stream-wrong-final':
        changed=payload[:44]+b'\\x00'*len(pcm)
        destination.write_bytes(changed)
        result['audio']['sha256']=hashlib.sha256(changed).hexdigest()
    final=json.dumps(result).encode()
    output.write(struct.pack('!cI',b'R',len(final))+final);output.flush()
else:
    print(json.dumps(result))
'''


@contextmanager
def prepared():
    fixture = fixtures.RuntimeTests("runTest")
    fixture.setUp()
    try:
        engine = fixture.runtime.python_root / "bin/python3.12"
        source = engine.read_text()
        assert source.count("print(json.dumps(result))") == 1
        engine.write_text(source.replace("print(json.dumps(result))", STREAM_TAIL))
        fixture.manifest["environment"]["python_sha256"] = hashlib.sha256(engine.read_bytes()).hexdigest()
        fixture.manifest["environment"]["python_root_sha256"] = tree_digest(fixture.runtime.python_root)
        fixture.start()
        yield fixture
    finally:
        fixture.doCleanups()


class ProcessStreamTests(unittest.TestCase):
    def test_stream_output_tuple_refuses_before_connect(self):
        with prepared() as fixture:
            for field, value in (("sample_rate_hz", 48_000), ("sample_format", "f32le")):
                args = fixture.arguments()
                args["output"][field] = value
                request = request_value("submit", job_id="stream", args=args, stream=True)
                with self.assertRaises(ProtocolError) as error:
                    client_request(fixture.ipc / "absent", request)
                self.assertEqual(error.exception.code, "UNSUPPORTED_CAPABILITY")

    def test_chunks_arrive_while_owned_worker_is_busy_and_bind_final_result(self):
        with prepared() as fixture, fixture.audio.open("rb") as audio:
            chunks = []
            def consume(sequence, offset, pcm):
                status = client_request(fixture.ipc, request_value("status"))
                self.assertEqual(status["provider_state"], "busy")
                chunks.append((time.monotonic(), sequence, offset, pcm))
            value = request_value("submit", job_id="stream", args=fixture.arguments(), timeout=5, stream=True)
            result, wave = client_request(fixture.ipc, value, audio.fileno(), on_chunk=consume)
            self.assertEqual([row[1:3] for row in chunks], [(0, 0), (1, 1200)])
            self.assertGreater(chunks[1][0] - chunks[0][0], .08)
            self.assertEqual(b"".join(row[3] for row in chunks), wave[44:])
            self.assertEqual(result["duration_ms"], 100)
            self.assertEqual(client_request(fixture.ipc, request_value("status"))["provider_state"], "ready")
            self.assertEqual(list(fixture.jobs.iterdir()), [])

    def test_partial_stream_cancel_and_deadline_release_the_owned_job(self):
        for ending in ("cancel", "deadline"):
            with self.subTest(ending=ending), prepared() as fixture, fixture.audio.open("rb") as audio:
                seen = []
                def consume(*row):
                    seen.append(row)
                    if ending == "cancel":
                        client_request(fixture.ipc, request_value("cancel", job_id="stream"))
                value = request_value("submit", job_id="stream", args=fixture.arguments("stream-hold"),
                                      timeout=1 if ending == "deadline" else 5, stream=True)
                with self.assertRaises(ProtocolError) as error:
                    client_request(fixture.ipc, value, audio.fileno(), on_chunk=consume)
                self.assertEqual(error.exception.code, "CANCELED" if ending == "cancel" else "DEADLINE_EXCEEDED")
                self.assertTrue(seen)
                self.assertEqual(client_request(fixture.ipc, request_value("status"))["provider_state"], "ready")
                self.assertEqual(list(fixture.jobs.iterdir()), [])

    def test_truncation_and_valid_but_different_final_wave_refuse(self):
        for text in ("stream-truncated", "stream-wrong-final"):
            with self.subTest(text=text), prepared() as fixture, fixture.audio.open("rb") as audio:
                value = request_value("submit", job_id="stream", args=fixture.arguments(text), timeout=5, stream=True)
                with self.assertRaises(ProtocolError) as error:
                    client_request(fixture.ipc, value, audio.fileno())
                self.assertEqual(error.exception.code, "MALFORMED_WORKER_RESULT")
                self.assertEqual(client_request(fixture.ipc, request_value("status"))["provider_state"], "ready")
                self.assertEqual(list(fixture.jobs.iterdir()), [])

    def test_consumer_disconnect_after_first_pcm_reaps_the_worker(self):
        with prepared() as fixture, fixture.audio.open("rb") as audio:
            def closed(*_row):
                raise BrokenPipeError("synthetic consumer closed")
            request = request_value("submit", job_id="stream", args=fixture.arguments("stream-hold"),
                                    timeout=5, stream=True)
            with self.assertRaises(BrokenPipeError):
                client_request(fixture.ipc, request, audio.fileno(), on_chunk=closed)
            until = time.monotonic() + 5
            while client_request(fixture.ipc, request_value("status"))["provider_state"] != "ready":
                self.assertLess(time.monotonic(), until)
                time.sleep(.01)
            self.assertEqual(list(fixture.jobs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
