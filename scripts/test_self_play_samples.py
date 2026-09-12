import base64
import http.client
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from self_play_samples import list_game_samples, read_game_image, PNG_SIGNATURE
from experiment_dashboard import Dashboard, handler


class SamplesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name)
        self.root = self.parent/'selfplay-nucleus'
        self.run = self.root/'nucleus-p095'
        (self.run/'games').mkdir(parents=True)
        (self.run/'stats').mkdir()
        (self.run/'checkpoints/00000000').mkdir(parents=True)
        (self.root/'plan.json').write_text(json.dumps({'jobs':[{'id':'nucleus-p095'}]}))
        (self.run/'stats/00000000.json').write_text('{"epoch":0}')
        self.metadata = self.run/'checkpoints/00000000/metadata.json'
        self.metadata.write_text('{}')
        self.png = PNG_SIGNATURE + struct.pack('>I',13) + b'IHDR' + struct.pack('>II',770,380)
        (self.run/'games/00000000.00.png').write_bytes(self.png)
        (self.run/'games/00000001.00.png').write_bytes(self.png)

    def test_lists_only_complete_epochs_and_reads_exact_selected_sample(self):
        rows = list_game_samples(self.run, {1})
        self.assertEqual(rows, [{'epoch':1,'sample':0,'width':770,'height':380,'positions':2}])
        result = read_game_image(self.parent/'queue','nucleus-p095',1,0)
        self.assertEqual(base64.b64decode(result['data']), self.png)
        self.metadata.unlink()
        with self.assertRaises(ValueError):
            read_game_image(self.parent/'queue','nucleus-p095',1,0)

    def test_rejects_traversal_invalid_selection_and_symlink(self):
        for phase, epoch, sample in [('../nucleus-p095',1,0),('unknown',1,0),('nucleus-p095',0,0),
                                     ('nucleus-p095',1,-1),('nucleus-p095',True,0)]:
            with self.assertRaises(ValueError):
                read_game_image(self.parent/'queue',phase,epoch,sample)
        link = self.run/'games/00000000.01.png'
        link.symlink_to(self.run/'games/00000000.00.png')
        with self.assertRaises(ValueError):
            read_game_image(self.parent/'queue','nucleus-p095',1,1)

    def test_long_game_uses_compact_sheet_and_preserves_position_count(self):
        path = self.run/'games/00000000.00.png'
        path.write_bytes(PNG_SIGNATURE + struct.pack('>I',13) + b'IHDR' + struct.pack('>II',75650,380))
        tiled = PNG_SIGNATURE + struct.pack('>I',13) + b'IHDR' + struct.pack('>II',6230,5060)
        path.with_suffix('.tiles.png').write_bytes(tiled)
        rows = list_game_samples(self.run, {1})
        self.assertEqual(rows, [{'epoch':1,'sample':0,'width':6230,'height':5060,'positions':194,'columns':16}])
        actual = read_game_image(self.parent/'queue','nucleus-p095',1,0)
        self.assertEqual(base64.b64decode(actual['data']), tiled)
        path.with_suffix('.tiles.png').unlink()
        path.with_suffix('.tiles.png').symlink_to(path)
        with self.assertRaises(ValueError):
            read_game_image(self.parent/'queue','nucleus-p095',1,0)

    def test_http_png_head_cache_and_read_only(self):
        fixture = self
        class Bridge:
            calls = 0
            def call(self, method, **args):
                self.calls += 1
                self.last_method = method
                return read_game_image(fixture.parent/'queue', **args)
        bridge = Bridge()
        dashboard = Dashboard(bridge)
        dashboard.state['snapshot'] = {'phases':[{'id':'nucleus-p095','game_samples':list_game_samples(self.run,{1})}]}
        server = ThreadingHTTPServer(('127.0.0.1',0),handler(dashboard,self.parent))
        worker = threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        try:
            connection = http.client.HTTPConnection('127.0.0.1',server.server_port)
            uri = '/api/experiments/game-image?phase_id=nucleus-p095&epoch=1&sample=0'
            connection.request('GET',uri)
            response = connection.getresponse()
            self.assertEqual(response.status,200)
            self.assertEqual(response.getheader('Content-Type'),'image/png')
            self.assertEqual(response.read(),self.png)
            connection.request('HEAD',uri)
            response = connection.getresponse()
            self.assertEqual(response.status,200)
            self.assertEqual(response.read(),b'')
            self.assertEqual(bridge.calls,1)
            self.assertEqual(bridge.last_method,'game_image')
            connection.request('GET',uri.replace('sample=0','sample=99'))
            response = connection.getresponse()
            self.assertEqual(response.status,404)
            response.read()
            connection.request('POST',uri,body=b'')
            response = connection.getresponse()
            self.assertEqual(response.status,405)
            response.read()
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == '__main__':
    unittest.main()
