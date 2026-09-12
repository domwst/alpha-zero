"""Expose only completed self-play sample PNGs, without arbitrary filesystem access."""
import base64
import json
from pathlib import Path
import re
import struct

MAX_IMAGE_BYTES = 4_000_000
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def image_dimensions(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError('Game image unavailable')
    with path.open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b'IHDR':
        raise ValueError('Invalid game image')
    return struct.unpack('>II', header[16:24])


def image_metadata(path):
    width, height = image_dimensions(path)
    if height != 380 or (width+10) % 390 or not 1 <= (width+10)//390 <= 361:
        raise ValueError('Unsupported game image layout')
    positions = (width+10)//390
    tiled = path.with_suffix('.tiles.png')
    if tiled.exists():
        columns = min(positions, 16)
        width, height = image_dimensions(tiled)
        if (width, height) != (columns*390-10, ((positions+columns-1)//columns)*390-10):
            raise ValueError('Invalid tiled game layout')
        return {'width':width,'height':height,'positions':positions,'columns':columns}
    return {'width':width,'height':height,'positions':positions}


def list_game_samples(directory, completed_epochs):
    rows = []
    for path in sorted((directory/'games').glob('*.png')):
        match = re.fullmatch(r'(\d{8})\.(\d{2})\.png', path.name)
        if not match or int(match[1])+1 not in completed_epochs:
            continue
        try:
            rows.append({'epoch':int(match[1])+1,'sample':int(match[2]),**image_metadata(path)})
        except (ValueError, OSError):
            continue
    return rows


def read_game_image(queue, phase_id, epoch, sample):
    root = (Path(queue).parent/'selfplay-nucleus').resolve()
    if not isinstance(phase_id, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', phase_id):
        raise ValueError('Unknown experiment')
    if type(epoch) is not int or type(sample) is not int or not 1 <= epoch <= 100000000 or not 0 <= sample <= 99:
        raise ValueError('Invalid game selection')
    plan = json.loads((root/'plan.json').read_text())
    if phase_id not in [job['id'] for job in plan['jobs']]:
        raise ValueError('Unknown experiment')
    directory = root/phase_id
    path = directory/'games'/f'{epoch-1:08}.{sample:02}.png'
    if not path.resolve().is_relative_to(root):
        raise ValueError('Game image unavailable')
    if not (directory/'checkpoints'/f'{epoch-1:08}'/'metadata.json').is_file():
        raise ValueError('Epoch is not complete')
    stats = json.loads((directory/'stats'/f'{epoch-1:08}.json').read_text())
    if stats.get('epoch') != epoch-1:
        raise ValueError('Epoch statistics mismatch')
    metadata = image_metadata(path)
    if 'columns' in metadata:
        path = path.with_suffix('.tiles.png')
    with path.open('rb') as stream:
        data = stream.read(MAX_IMAGE_BYTES+1)
    if len(data)>MAX_IMAGE_BYTES:
        raise ValueError('Game image exceeds size limit')
    return {**metadata,'data':base64.b64encode(data).decode('ascii')}
