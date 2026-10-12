"""Version-1 replay storage: named MessagePack maps in Zstd, with JSON fallback.

Only native f32 fields use single precision. Duration and unknown fields retain
their types/precision. The HTTP representation is unchanged. Manifests stay JSON.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import struct
import tempfile
from array import array
from pathlib import Path

import msgpack
import zstandard as zstd

SUFFIX = ".msgpack.zst"
# A corrupt frame must not allocate unbounded memory in the public reader.
MAX_REPLAY_BYTES = 128 * 1024 * 1024
GAME_ID = re.compile(r"[0-9a-f]{1,64}")


def game_id(path):
    name = Path(path).name
    for suffix in (SUFFIX, ".json"):
        if name.endswith(suffix) and GAME_ID.fullmatch(name[: -len(suffix)]):
            return name[: -len(suffix)]
    return None


def find_game(directory, identifier):
    if not GAME_ID.fullmatch(identifier):
        raise ValueError("Invalid game identity")
    for suffix in (SUFFIX, ".json"):
        path = Path(directory) / (identifier + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No replay {identifier} in {directory}")


def game_paths(directory):
    """One entry per ID during mixed-format migration; prefer the new format."""
    paths = {}
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                identifier = game_id(entry.name)
                if (
                    identifier is not None
                    and entry.is_file()
                    and (identifier not in paths or entry.name.endswith(SUFFIX))
                ):
                    paths[identifier] = Path(entry.path)
    except FileNotFoundError:
        pass
    return paths.values()


def has_games(directory):
    try:
        with os.scandir(directory) as entries:
            return any(
                game_id(entry.name) is not None and entry.is_file() for entry in entries
            )
    except FileNotFoundError:
        return False


def f32(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected finite f32 replay value")
    result = struct.unpack(">f", struct.pack(">f", value))[0]
    if not math.isfinite(result):
        raise ValueError("Replay value exceeds f32 range")
    return result


def normalize(game):
    """Copy modified branches and restore f32 semantics; never mutate the input."""
    if game.get("schema_version") != 1:
        raise ValueError("Unsupported replay schema")
    result = dict(game)
    record = result["record"] = dict(game["record"])
    if "terminal_value" in record:
        record["terminal_value"] = f32(record["terminal_value"])
    record["plies"] = [dict(ply) for ply in record["plies"]]
    for ply in record["plies"]:
        if "value_actual" in ply:
            ply["value_actual"] = f32(ply["value_actual"])
        decision = dict(ply.get("decision") or {})
        if isinstance(ply.get("decision"), dict):
            ply["decision"] = decision
        diagnostics = dict(decision.get("diagnostics") or {})
        if isinstance(decision.get("diagnostics"), dict):
            decision["diagnostics"] = diagnostics
        for container, key in (
            (decision, "training_policy"),
            (diagnostics, "network_prior"),
            (diagnostics, "sampling_policy"),
        ):
            if container.get(key) is not None:
                values = container[key]
                if not all(type(v) in (int, float) for v in values):
                    raise ValueError("Expected numeric f32 vector")
                # Bulk conversion avoids per-element struct pack/unpack calls.
                converted = array("f", values).tolist()
                if not all(map(math.isfinite, converted)):
                    raise ValueError("Expected finite f32 replay values")
                container[key] = converted
        for key in ("value_estimate", "search_value"):
            if key in diagnostics:
                diagnostics[key] = f32(diagnostics[key])
    return result


def pack(game):
    """Encode maps compatible with rmp-serde::Serializer::with_struct_map()."""
    wide = msgpack.Packer(use_bin_type=True)
    single = msgpack.Packer(use_bin_type=True, use_single_float=True)
    output = io.BytesIO()

    def mapping(value, kind):
        output.write(wide.pack_map_header(len(value)))
        for key, item in value.items():
            output.write(wide.pack(key))
            if (
                (kind == "record" and key == "terminal_value")
                or (kind == "ply" and key == "value_actual")
                or (kind == "decision" and key == "training_policy")
                or (
                    kind == "diagnostics"
                    and key
                    in (
                        "network_prior",
                        "sampling_policy",
                        "value_estimate",
                        "search_value",
                    )
                )
            ):
                output.write(single.pack(item))
            elif kind == "record" and key == "plies":
                output.write(wide.pack_array_header(len(item)))
                for ply in item:
                    mapping(ply, "ply")
            elif (kind, key) in (
                ("game", "record"),
                ("ply", "decision"),
                ("decision", "diagnostics"),
            ) and isinstance(item, dict):
                mapping(item, key)
            else:
                output.write(wide.pack(item))

    mapping(game, "game")
    payload = output.getvalue()
    if len(payload) > MAX_REPLAY_BYTES:
        raise ValueError("Replay exceeds size limit")
    return payload


def unpack(compressed):
    size = zstd.frame_content_size(compressed)
    if (
        size not in (zstd.CONTENTSIZE_UNKNOWN, zstd.CONTENTSIZE_ERROR)
        and size > MAX_REPLAY_BYTES
    ):
        raise ValueError("Replay exceeds size limit")
    payload = zstd.ZstdDecompressor().decompress(
        compressed,
        max_output_size=MAX_REPLAY_BYTES,
        allow_extra_data=False,
    )
    if len(payload) > MAX_REPLAY_BYTES:
        raise ValueError("Replay exceeds size limit")
    game = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    if not isinstance(game, dict) or game.get("schema_version") != 1:
        raise ValueError("Unsupported replay schema")
    return game


def load(path):
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        # A list request may race atomic conversion/removal of its JSON entry.
        if path.suffix != ".json" or game_id(path) is None:
            raise
        return load(find_game(path.parent, game_id(path)))
    if path.name.endswith(SUFFIX):
        return unpack(raw)
    return json.loads(raw)


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(directory, game):
    """Publish only after exact native-semantic round-trip verification."""
    directory = Path(directory)
    identifier = game["game_id"]
    if not GAME_ID.fullmatch(identifier):
        raise ValueError("Invalid game identity")
    expected = normalize(game)
    target = directory / (identifier + SUFFIX)
    directory.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if load(target) != expected:
            raise ValueError("Existing compressed replay differs")
        return target
    encoded = zstd.ZstdCompressor(level=6, write_checksum=True).compress(pack(expected))
    if unpack(encoded) != expected:
        raise ValueError("Replay round-trip mismatch")
    fd, name = tempfile.mkstemp(prefix=identifier + ".", suffix=".tmp", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        # Never overwrite an existing publication, even with concurrent importers.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if load(target) != expected:
                raise ValueError("Concurrent compressed replay differs")
        sync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return target
