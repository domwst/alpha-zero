import copy
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import msgpack
import zstandard as zstd

from scripts.job_service import compress_replays, replays


def fixture():
    return {
        "schema_version": 1,
        "game_type": "gomoku",
        "game_id": "00000001",
        "seed": 2**64 - 1,
        "model_identity": None,
        "models_by_seat": {},
        "duration_seconds": 1.2345678901234567,
        "policy_semantics": "visits",
        "provenance": "test",
        "record": {
            "terminal_state": {"state": [0] * 91},
            "terminal_actor": "Second",
            "terminal_value": 0.12345678,
            "plies": [
                {
                    "state": {"state": [0] * 91},
                    "action": {"x": 0, "y": 0},
                    "actor": "First",
                    "turn_change": "SwitchPlayer",
                    "decision": {
                        "move_index": 0,
                        "training_policy": [0.12345678, 0.8765432],
                        "diagnostics": {
                            "network_prior": [0.1, 0.9],
                            "sampling_policy": None,
                            "value_estimate": 0.12345678,
                            "search_value": None,
                            "root_visits": [1, 4294967295],
                            "search": None,
                        },
                    },
                }
            ],
        },
    }


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_roundtrip_precision_and_unknown_metadata(self):
        game = fixture()
        game["unknown"] = {"number": 1.2345678901234567}
        game["record"]["plies"][0]["value_actual"] = 0.12345678
        before = copy.deepcopy(game)
        path = replays.save(self.root, game)
        restored = replays.load(path)
        self.assertEqual(restored, replays.normalize(game))
        self.assertEqual(game, before)
        self.assertEqual(restored["duration_seconds"], game["duration_seconds"])
        self.assertEqual(restored["unknown"], game["unknown"])
        self.assertEqual(restored["seed"], 2**64 - 1)
        self.assertTrue(zstd.get_frame_parameters(path.read_bytes()).has_checksum)

    def test_legacy_and_mixed_directory(self):
        game = fixture()
        old = self.root / (game["game_id"] + ".json")
        old.write_text(json.dumps(game))
        (self.root / "receipt.json").write_text("{}")
        self.assertEqual(replays.load(old), game)
        new = replays.save(self.root, game)
        self.assertEqual(list(replays.game_paths(self.root)), [new])
        self.assertEqual(replays.find_game(self.root, game["game_id"]), new)
        old.unlink()
        self.assertEqual(replays.load(old), replays.normalize(game))
        self.assertEqual(replays.save(self.root, game), new)

    def test_conflict_and_invalid_id(self):
        game = fixture()
        path = replays.save(self.root, game)
        original = path.read_bytes()
        game["seed"] = 0
        with self.assertRaises(ValueError):
            replays.save(self.root, game)
        self.assertEqual(path.read_bytes(), original)
        game["game_id"] = "../escape"
        with self.assertRaises(ValueError):
            replays.save(self.root, game)

    def test_bulk_float_conversion_and_imported_nulls(self):
        game = fixture()
        game["seed"] = None  # Imported archives are browsable, not resumable.
        game["record"]["terminal_state"] = None
        ply = game["record"]["plies"][0]
        ply["action"] = None
        values = [0, -0.0, 1.0, -1.0, 1e-40, 1e-46, 0.12345678]
        ply["decision"]["training_policy"] = values
        normalized = replays.normalize(game)
        self.assertEqual(
            normalized["record"]["plies"][0]["decision"]["training_policy"],
            [replays.f32(v) for v in values],
        )
        self.assertEqual(replays.load(replays.save(self.root, game)), normalized)
        for value in (float("nan"), float("inf"), 1e100, True, "0.5", None):
            ply["decision"]["training_policy"] = [value]
            with self.assertRaises((ValueError, OverflowError)):
                replays.normalize(game)

    def test_corruption_limits_and_schema(self):
        compressed = replays.save(self.root, fixture()).read_bytes()
        corrupted = bytearray(compressed)
        corrupted[-1] ^= 1
        for data in (compressed[:-1], bytes(corrupted), compressed + b"extra"):
            with self.assertRaises((ValueError, zstd.ZstdError)):
                replays.unpack(data)
        with (
            patch.object(replays, "MAX_REPLAY_BYTES", 32),
            self.assertRaises(ValueError),
        ):
            replays.unpack(compressed)
        for data in ({"schema_version": 2}, [1, 2]):
            with self.assertRaises(ValueError):
                replays.unpack(zstd.ZstdCompressor().compress(msgpack.packb(data)))

    def test_migration_resume_aliases_and_durable_removal(self):
        directory = self.root / "job/games/archive/00000000"
        directory.mkdir(parents=True)
        game = fixture()
        original = directory / (game["game_id"] + ".json")
        original.write_text(json.dumps(game))
        alias_directory = self.root / "second/games/archive/00000000"
        alias_directory.mkdir(parents=True)
        alias = alias_directory / original.name
        os.link(original, alias)
        (self.root / "linked-job").symlink_to(
            directory.parent.parent.parent, target_is_directory=True
        )
        manifest = directory / "manifest.json"
        manifest.write_text("{}")
        groups = compress_replays.inventory(self.root)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        first = compress_replays.prepare(groups[0])
        second = compress_replays.prepare(groups[0])
        self.assertEqual(first, second)
        targets = [Path(p) for p in second["targets"]]
        self.assertEqual(targets[0].stat().st_ino, targets[1].stat().st_ino)
        with (self.root / "journal").open("w+") as journal:
            compress_replays.commit([second], journal, True)
            journal.seek(0)
            self.assertEqual(json.load(journal), second)
        self.assertFalse(original.exists())
        self.assertFalse(alias.exists())
        self.assertTrue(manifest.exists())
        self.assertEqual(compress_replays.inventory(self.root), [])
        self.assertEqual(replays.load(targets[0]), replays.normalize(game))

    def test_migration_refuses_changed_source(self):
        game = fixture()
        original = self.root / (game["game_id"] + ".json")
        original.write_text(json.dumps(game))
        entry = compress_replays.prepare([str(original)])
        original.write_text("changed")
        with (
            (self.root / "journal").open("w") as journal,
            self.assertRaises(ValueError),
        ):
            compress_replays.commit([entry], journal, True)
        self.assertEqual(original.read_text(), "changed")

    @unittest.skipUnless(
        os.environ.get("ALZ_REPLAY_CODEC"),
        "set ALZ_REPLAY_CODEC for Rust interoperability",
    )
    def test_rust_interoperability(self):
        game = fixture()
        legacy = self.root / (game["game_id"] + ".json")
        legacy.write_text(json.dumps(game))
        compressed = replays.save(self.root, game)
        for source in (legacy, compressed):
            destination = self.root / source.suffix[1:]
            subprocess.run(
                [
                    os.environ["ALZ_REPLAY_CODEC"],
                    str(source),
                    str(destination),
                    str(game["seed"]),
                ],
                check=True,
            )
            restored = replays.load(replays.find_game(destination, game["game_id"]))
            self.assertEqual(restored, replays.normalize(game))
            payload = zstd.ZstdDecompressor().decompress(
                replays.find_game(destination, game["game_id"]).read_bytes()
            )
            self.assertIn(b"\xca" + struct.pack(">f", 0.12345678), payload)
            self.assertIn(
                b"\xcb" + struct.pack(">d", game["duration_seconds"]), payload
            )
        result = subprocess.run(
            [
                os.environ["ALZ_REPLAY_CODEC"],
                str(compressed),
                str(self.root / "bad"),
                "1",
            ],
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        # Native recovery must reject corruption and trailing MessagePack objects.
        encoded = compressed.read_bytes()
        payload = zstd.ZstdDecompressor().decompress(encoded)
        for damaged in (
            encoded[:-1],
            zstd.ZstdCompressor().compress(payload + b"\xc0"),
        ):
            bad = self.root / ("bad" + replays.SUFFIX)
            bad.write_bytes(damaged)
            result = subprocess.run(
                [
                    os.environ["ALZ_REPLAY_CODEC"],
                    str(bad),
                    str(self.root / "bad-output"),
                    str(game["seed"]),
                ],
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
