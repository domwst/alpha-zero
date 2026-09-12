#!/usr/bin/env python3
"""Build a checksummed archive of experiment data, deduplicating identical files."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import time

EXCLUDE = {'.cache', '.cargo', '.rustup', '.local', '.venv', '.git', '.ssh',
           'target', 'node_modules', '__pycache__', '.env'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(output):
    paths, excluded = [], []
    for directory, dirs, files in os.walk('/workspace', followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            if name in EXCLUDE or path.is_symlink() or str(path) == '/workspace/alz-training-profile-20260906/tools':
                dirs.remove(name)
                excluded.append({'path': str(path), 'reason': 'rebuildable dependency/cache or directory symlink',
                                 'target': os.readlink(path) if path.is_symlink() else None})
        for name in files:
            path = parent / name
            if name in EXCLUDE or name.startswith('.env.') or path.is_symlink():
                excluded.append({'path': str(path), 'reason': 'credential/config secret or symlink',
                                 'target': os.readlink(path) if path.is_symlink() else None})
            elif path.is_file(): paths.append(path)
    # Preserve disposable diagnostic outputs; build caches and transfer encodings are reproducible.
    paths.extend(p for p in Path('/tmp').iterdir() if p.is_file() and not p.is_symlink()
                 and not p.name.startswith('alz-decommission')
                 and p.suffix in ('.log', '.json') and p.name.startswith(('alz', 'capacity', 'global', 'training')))
    entries, content, unique_bytes = [], {}, 0
    started = time.time()
    with tarfile.open(output, 'w:gz', compresslevel=3) as archive:
        for index, path in enumerate(sorted(paths)):
            before = path.stat()
            checksum = sha256(path)
            name = str(path).lstrip('/')
            info = archive.gettarinfo(str(path), arcname=name)
            key = (before.st_size, checksum)
            if key in content:
                info.type = tarfile.LNKTYPE
                info.linkname = content[key]
                info.size = 0
                archive.addfile(info)
            else:
                with path.open('rb') as stream: archive.addfile(info, stream)
                content[key] = name
                unique_bytes += before.st_size
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Source changed during archive: ' + str(path))
            entries.append({'path': name, 'size': before.st_size, 'sha256': checksum,
                            'mode': stat.S_IMODE(before.st_mode)})
            if index % 100 == 0: print(json.dumps({'files': index, 'unique_bytes': unique_bytes}), flush=True)
        manifest = {'schema_version': 1, 'files': entries, 'excluded': excluded,
                    'logical_bytes': sum(e['size'] for e in entries), 'unique_bytes': unique_bytes,
                    'unique_contents': len(content), 'created_at': time.time()}
        encoded = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('MANIFEST.json'); info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    output.with_suffix('.manifest.json').write_bytes(encoded)
    receipt = {'archive': str(output), 'archive_bytes': output.stat().st_size,
               'archive_sha256': sha256(output), 'files': len(entries),
               'unique_contents': len(content), 'unique_bytes': unique_bytes,
               'logical_bytes': manifest['logical_bytes'], 'seconds': time.time() - started}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
