import json
from pathlib import Path
import tempfile
import unittest
from stop_self_play_at_epoch import complete_epochs


class EpochBoundaryTests(unittest.TestCase):
    def test_requires_complete_stats_and_matching_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'stats').mkdir()
            (root/'checkpoints/00000000').mkdir(parents=True)
            (root/'stats/00000000.json').write_text('{"epoch":')
            (root/'checkpoints/00000000/metadata.json').write_text('{}')
            self.assertEqual(complete_epochs(root), [])
            (root/'stats/00000000.json').write_text(json.dumps({'epoch':0}))
            (root/'stats/00000001.json').write_text(json.dumps({'epoch':1}))
            self.assertEqual(complete_epochs(root), [0])
            (root/'checkpoints/00000001').mkdir()
            (root/'checkpoints/00000001/metadata.json').write_text('{}')
            self.assertEqual(complete_epochs(root), [0,1])


if __name__=='__main__':
    unittest.main()
