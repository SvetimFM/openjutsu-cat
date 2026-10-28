// KinectBurst.cs - Kinect v2 depth burst recorder (Kinect for Windows SDK 2.0).
//
// Opens the sensor ONCE, warms up, and writes burst-ready.txt only after a real
// depth frame is in hand - that file is the "GO" signal: never cue a performer
// before it exists. Then captures timestamped frames at a fixed cadence.
//
// Outputs (in the working directory):
//   burst-ready.txt   written after the warmup depth frame + baseline are saved
//   burst-log.csv     index,unix_ms,min_mm,mean_mm,valid_px per frame
//   f000.bin          raw depth, 512x424 uint16 LE, millimetres (baseline frame:
//                     the room as it looked when armed)
//   f001.bin ...      burst frames
//   fNNN.png          8-bit viz of the same frame (native res, no upscale)
//   burst-done.txt    written when the burst completes cleanly
//
// Usage: KinectBurst.exe [frames=40] [interval_ms=500]
//
// Timing: each tick waits interval_ms for the newest depth frame, THEN writes
// it (.bin + .png, ~30-50 ms), so the real interval is interval_ms + write time;
// a tick that sees no new frame records nothing. Measured on a Kinect v2 +
// SATA SSD: interval_ms=28 -> ~63 ms median (~15 fps, 400/400 frames), while
// interval_ms=62 -> ~94 ms (~10.6 fps). Use 28 for the 16 fps control spec;
// capture/encode_depth.py resamples by the logged timestamps either way.
// Build: capture/build_kinectburst.ps1
using System;
using System.IO;
using System.Threading;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using Microsoft.Kinect;

class KinectBurst {
    static void SaveViz(ushort[] depth, int w, int h, string path) {
        var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
        var px = new byte[w * h * 4];
        for (int i = 0; i < depth.Length; i++) {
            int d = depth[i];
            byte v = d == 0 ? (byte)0 : (byte)Math.Max(0, Math.Min(255, 255 - (d - 500) * 255 / 4000));
            px[i * 4] = v; px[i * 4 + 1] = v; px[i * 4 + 2] = v; px[i * 4 + 3] = 255;
        }
        var data = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
        Marshal.Copy(px, 0, data.Scan0, px.Length);
        bmp.UnlockBits(data);
        bmp.Save(path, ImageFormat.Png);
        bmp.Dispose();
    }

    static void SaveRaw(ushort[] depth, string path) {
        var bytes = new byte[depth.Length * 2];
        Buffer.BlockCopy(depth, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    static void Main(string[] args) {
        int frames = args.Length > 0 ? int.Parse(args[0]) : 40;
        int intervalMs = args.Length > 1 ? int.Parse(args[1]) : 500;

        var sensor = KinectSensor.GetDefault();
        if (sensor == null) { Console.WriteLine("NOSENSOR"); return; }
        sensor.Open();
        var reader = sensor.OpenMultiSourceFrameReader(FrameSourceTypes.Depth);
        try {
            // Warmup: poll until the first real depth frame arrives.
            ushort[] buf = null; int w = 0, h = 0;
            var deadline = DateTime.Now.AddSeconds(30);
            while (DateTime.Now < deadline && buf == null) {
                var msf = reader.AcquireLatestFrame();
                if (msf == null) { Thread.Sleep(50); continue; }
                var df = msf.DepthFrameReference.AcquireFrame();
                if (df != null) using (df) {
                    w = df.FrameDescription.Width; h = df.FrameDescription.Height;
                    buf = new ushort[w * h];
                    df.CopyFrameDataToArray(buf);
                }
            }
            if (buf == null) { Console.WriteLine("NODEPTH"); return; }
            Console.WriteLine("SENSOR " + w + "x" + h);

            using (var log = new StreamWriter("burst-log.csv")) {
                log.WriteLine("index,unix_ms,min_mm,mean_mm,valid_px");
                Action<int, ushort[]> record = (idx, d) => {
                    long ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    int min = int.MaxValue; long sum = 0; int valid = 0;
                    for (int i = 0; i < d.Length; i++) {
                        if (d[i] > 0) { if (d[i] < min) min = d[i]; sum += d[i]; valid++; }
                    }
                    log.WriteLine(idx + "," + ms + "," + (valid > 0 ? min : 0) + "," +
                                  (valid > 0 ? sum / valid : 0) + "," + valid);
                    log.Flush();
                    SaveRaw(d, "f" + idx.ToString("000") + ".bin");
                    SaveViz(d, w, h, "f" + idx.ToString("000") + ".png");
                };

                record(0, buf); // baseline: whatever the room looks like at arm time
                File.WriteAllText("burst-ready.txt",
                    DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString());
                Console.WriteLine("READY");

                for (int i = 1; i <= frames; i++) {
                    var tickEnd = DateTime.Now.AddMilliseconds(intervalMs);
                    ushort[] latest = null;
                    while (DateTime.Now < tickEnd) {
                        var msf = reader.AcquireLatestFrame();
                        if (msf == null) { Thread.Sleep(20); continue; }
                        var df = msf.DepthFrameReference.AcquireFrame();
                        if (df != null) using (df) {
                            if (latest == null) latest = new ushort[w * h];
                            df.CopyFrameDataToArray(latest);
                        }
                    }
                    if (latest != null) record(i, latest);
                }
            }
            File.WriteAllText("burst-done.txt",
                DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString());
            Console.WriteLine("DONE");
        } finally {
            reader.Dispose();
            sensor.Close();
        }
    }
}
