"""Every script must start under ComfyUI's Windows portable python, whose
._pth file keeps the script's own folder off sys.path. `python -I` (isolated
mode) reproduces that on any OS."""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["run_local.py", "camera_slate.py", "tools/validate_workflows.py", "tools/verify_models.py"]


class IsolatedStart(unittest.TestCase):
    def test_scripts_start_without_their_folder_on_sys_path(self):
        for script in SCRIPTS:
            if not os.path.exists(os.path.join(ROOT, script)):
                continue
            # run from another directory so the cwd can't help either
            r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, script), "--help"],
                               cwd=os.path.dirname(ROOT), capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, f"{script}:\n{r.stderr[-1500:]}")
            self.assertIn("usage", r.stdout.lower(), script)


if __name__ == "__main__":
    unittest.main()
