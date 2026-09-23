"""Exercise real SSE parsing with speculative batches and delayed trailers."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('long_context', Path(__file__).resolve().parents[1] / 'tools/validate-long-context.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

class StreamTiming(unittest.TestCase):
    def run_stream(self, first_batch, usage_count=10):
        clock = [0.0]
        tokens = list(range(10))
        events = [
            (2.0, {'choices': [{'text': 'a', 'token_ids': tokens[:first_batch]}]}),
            (5.0, {'choices': [{'text': 'b', 'token_ids': tokens[first_batch:], 'finish_reason': 'length'}]}),
            (12.0, {'choices': [], 'usage': {'completion_tokens': usage_count}}),
        ]
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                for stamp, data in events:
                    clock[0] = stamp
                    yield b'data: ' + json.dumps(data).encode() + b'\n'
                yield b'data: [DONE]\n'
        client = protocol.Client('http://unused', 'synthetic', 'model')
        with patch.object(client, 'request', return_value=Response()), \
             patch.object(protocol, 'available_memory', return_value=100 * 1024**3), \
             patch.object(protocol.time, 'monotonic', side_effect=lambda: clock[0]):
            return client.generate([1, 2], 10, 100, 0)

    def test_first_speculative_batch_and_usage_trailer_are_excluded(self):
        result = self.run_stream(4)
        self.assertEqual(result['ttftSeconds'], 2.0)
        self.assertEqual(result['firstEmissionTokens'], 4)
        self.assertEqual(result['decodeWindowSeconds'], 3.0)
        self.assertEqual((len(result['tokenIds']) - result['firstEmissionTokens']) / result['decodeWindowSeconds'], 2.0)
        self.assertEqual(result['elapsedSeconds'], 12.0)

    def test_single_token_first_chunk_uses_same_window(self):
        result = self.run_stream(1)
        self.assertEqual(result['firstEmissionTokens'], 1)
        self.assertEqual(result['decodeWindowSeconds'], 3.0)

    def test_inconsistent_usage_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'count differs'):
            self.run_stream(4, usage_count=9)

if __name__ == '__main__': unittest.main()
