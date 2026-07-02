"""
Aggregate REAL on-device latency from the app's [TIMING] logcat lines into a
mean +/- std table for the Testing chapter.

HOW TO GET THE DATA (must be a real phone, not an emulator, for honest numbers):
  1. Build & run the app on the device:  flutter run --release
  2. In the app, analyze ~10 varied images (different conditions + a normal one).
  3. Capture logcat to a file, e.g.:
         adb logcat -d | grep "\[TIMING\]" > timing.log
     (or copy the [TIMING] lines from `flutter run` output into timing.log)
  4. Run:  python parse_timing.py timing.log

It parses lines like:
  [TIMING] preprocess=42ms gate=88ms cnn=510ms scorecam=1320ms total=1960ms | Tinea
"""
import sys, re, statistics as st

path = sys.argv[1] if len(sys.argv) > 1 else "timing.log"
pat = re.compile(r"preprocess=(\d+)ms\s+gate=(\d+)ms\s+cnn=(\d+)ms\s+scorecam=(\d+)ms\s+total=(\d+)ms")
stages = ["preprocess", "gate", "cnn", "scorecam", "total"]
data = {s: [] for s in stages}

with open(path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            for s, v in zip(stages, m.groups()):
                data[s].append(int(v))

n = len(data["total"])
if n == 0:
    print(f"No [TIMING] lines found in {path}. Check the capture step.")
    sys.exit(1)

print(f"\nParsed {n} analysis runs from {path}\n")
print(f"{'Stage':>12}{'mean (ms)':>12}{'std (ms)':>11}{'min':>8}{'max':>8}")
for s in stages:
    vals = data[s]
    mean = st.mean(vals)
    sd = st.pstdev(vals) if n > 1 else 0.0
    print(f"{s:>12}{mean:>12.0f}{sd:>11.0f}{min(vals):>8}{max(vals):>8}")

# Ready-to-paste LaTeX rows
print("\n--- LaTeX table rows (paste into Testing.tex) ---")
label = {"preprocess": "Image preprocessing", "gate": "Normal-vs-Disease gate",
         "cnn": "B2+B3 ensemble", "scorecam": "Score-CAM", "total": "End-to-end total"}
for s in stages:
    vals = data[s]
    mean = st.mean(vals)
    sd = st.pstdev(vals) if n > 1 else 0.0
    print(f"{label[s]} & {mean:.0f} $\\pm$ {sd:.0f} \\\\ \\hline")
print(f"\n(n = {n} runs)")
