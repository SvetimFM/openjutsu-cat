"""capture/capture_take.ps1 guard rails, driven with fake recorders (Windows only).
A real-sensor smoke test runs when OPENJUTSU_KINECTBURST points at KinectBurst.exe."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "capture", "capture_take.ps1")

FAKE_OK = r"""@echo off
echo SENSOR 512x424
>burst-log.csv echo index,unix_ms,min_mm,mean_mm,valid_px
>>burst-log.csv echo 0,1000,0,0,0
type nul > f000.bin
>burst-ready.txt echo 1000
echo READY
>>burst-log.csv echo 1,1063,0,0,0
type nul > f001.bin
>>burst-log.csv echo 2,1126,0,0,0
type nul > f002.bin
>>burst-log.csv echo 3,1500,0,0,0
type nul > f003.bin
>burst-done.txt echo 1500
echo DONE
"""
FAKE_NOSENSOR = "@echo off\r\necho NOSENSOR\r\n"
FAKE_HANG = "@echo off\r\n>burst-ready.txt echo 1\r\nping -n 30 127.0.0.1 >nul\r\n"


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell"), "Windows PowerShell only")
class CaptureTake(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake(self, body, name="fake.cmd"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", newline="\r\n") as f:
            f.write(body)
        return path

    def run_ps(self, take, *extra):
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", SCRIPT,
                            "-TakeDir", take, "-Quiet", *extra], capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def status(self, take):
        with open(os.path.join(take, "capture-status.txt")) as f:
            return f.read().strip()

    def test_existing_take_folder_is_never_reused(self):
        take = os.path.join(self.tmp, "luna-01")
        os.makedirs(take)
        open(os.path.join(take, "f001.bin"), "w").close()
        rc, out = self.run_ps(take, "-Exe", self.fake(FAKE_OK))
        self.assertEqual(rc, 3, out)
        self.assertEqual(os.listdir(take), ["f001.bin"])

    def test_refuses_while_another_program_owns_the_sensor(self):
        me = os.path.splitext(os.path.basename(sys.executable))[0]  # certainly running
        take = os.path.join(self.tmp, "t")
        rc, out = self.run_ps(take, "-Exe", self.fake(FAKE_OK), "-SensorOwners", me)
        self.assertEqual(rc, 2, out)
        self.assertFalse(os.path.exists(take))

    def test_missing_recorder(self):
        rc, out = self.run_ps(os.path.join(self.tmp, "t"), "-Exe", os.path.join(self.tmp, "nope.exe"))
        self.assertEqual(rc, 4, out)

    def test_recorder_that_never_sees_the_sensor_fails_with_its_message(self):
        take = os.path.join(self.tmp, "t")
        rc, out = self.run_ps(take, "-Exe", self.fake(FAKE_NOSENSOR))
        self.assertEqual(rc, 6, out)
        self.assertIn("NOSENSOR", self.status(take))

    def test_ready_file_alone_is_not_go(self):
        # burst-ready.txt but no frame ever lands: must time out, not say GO
        take = os.path.join(self.tmp, "t")
        rc, out = self.run_ps(take, "-Exe", self.fake(FAKE_HANG), "-ReadyTimeoutSec", "3")
        self.assertEqual(rc, 6, out)
        self.assertNotIn("GO ", out)
        self.assertTrue(self.status(take).startswith("FAIL no depth frames"))

    def test_clean_take_reports_rate_and_stalls(self):
        take = os.path.join(self.tmp, "t")
        rc, out = self.run_ps(take, "-Exe", self.fake(FAKE_OK), "-Frames", "3")
        self.assertEqual(rc, 0, out)
        self.assertIn("GO ", out)
        st = self.status(take)
        self.assertTrue(st.startswith("DONE recorded 3/3"), st)
        self.assertIn("stalls>150ms: 374 ms", st)  # 1126 -> 1500

    @unittest.skipUnless(os.environ.get("OPENJUTSU_KINECTBURST"), "set OPENJUTSU_KINECTBURST for the sensor test")
    def test_real_sensor_short_take(self):
        take = os.path.join(os.environ.get("OPENJUTSU_TAKE_ROOT", self.tmp), "smoke-" + os.urandom(3).hex())
        rc, out = self.run_ps(take, "-Exe", os.environ["OPENJUTSU_KINECTBURST"], "-Frames", "40")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.status(take).startswith("DONE recorded"), self.status(take))


if __name__ == "__main__":
    unittest.main()
