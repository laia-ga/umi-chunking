import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_indexing


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.cache_dir_patch = patch.object(
            run_indexing,
            "EMBEDDINGS_CACHE_DIR",
            self.cache_dir,
        )
        self.cache_dir_patch.start()
        self.addCleanup(self.cache_dir_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def test_completed_batches_are_cached_and_reused_after_interruption(self):
        chunks = [
            {"chunk_id": "chunk-1", "text": "first"},
            {"chunk_id": "chunk-2", "text": "second"},
            {"chunk_id": "chunk-3", "text": "third"},
        ]
        embeddings = SimpleNamespace(encode_kwargs={"normalize_embeddings": False})

        def interrupted_batches(**kwargs):
            self.assertEqual(kwargs["cache_batch_size"], 2)
            yield 0, [[1.0, 0.0], [0.0, 1.0]]
            raise RuntimeError("simulated interruption")

        with patch.object(
            run_indexing,
            "iter_embed_text_batches",
            side_effect=interrupted_batches,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_indexing.embed_with_cache(
                    embeddings=embeddings,
                    chunks=chunks,
                    model_name="test-model",
                    num_workers=1,
                    batch_size=2,
                    cache_batch_size=2,
                    gpu_ids=None,
                    cpu_threads=1,
                    use_cache=True,
                )

        cache = run_indexing.load_embeddings_cache("test-model")
        self.assertEqual(set(cache), {"chunk-1", "chunk-2"})

        def remaining_batch(**kwargs):
            self.assertEqual(kwargs["texts"], ["third"])
            yield 0, [[0.5, 0.5]]

        with patch.object(
            run_indexing,
            "iter_embed_text_batches",
            side_effect=remaining_batch,
        ):
            vectors, stats = run_indexing.embed_with_cache(
                embeddings=embeddings,
                chunks=chunks,
                model_name="test-model",
                num_workers=1,
                batch_size=2,
                cache_batch_size=2,
                gpu_ids=None,
                cpu_threads=1,
                use_cache=True,
            )

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        self.assertEqual(stats["cache_hits"], 2)
        self.assertEqual(stats["cache_misses"], 1)

    def test_incomplete_final_cache_line_is_discarded(self):
        cache_file = run_indexing.embeddings_cache_path("test-model")
        cache_file.write_bytes(
            (
                json.dumps(
                    {
                        "chunk_id": "valid",
                        "text_hash": "hash",
                        "vector": [1.0, 0.0],
                    }
                )
                + "\n"
                + '{"chunk_id":"partial"'
            ).encode("utf-8")
        )

        cache = run_indexing.load_embeddings_cache("test-model")

        self.assertEqual(set(cache), {"valid"})
        self.assertTrue(cache_file.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
