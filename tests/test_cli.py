import json
import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_status_reports_runnable_backend(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "city_twin", "status"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout)["status"], "ready")


if __name__ == "__main__":
    unittest.main()
