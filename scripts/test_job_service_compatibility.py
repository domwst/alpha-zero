"""The compute-host service must parse on its documented minimum Python version."""
import ast
from pathlib import Path
import unittest

class CompatibilityTests(unittest.TestCase):
    def test_service_syntax_supports_python_312(self):
        for path in (Path(__file__).parent / 'job_service').glob('*.py'):
            with self.subTest(module=path.name):
                ast.parse(path.read_text(), filename=str(path), feature_version=(3, 12))
