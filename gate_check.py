import sys, time
from bilm import BILM
from bilm.metrics import bits_per_byte
import json
from pathlib import Path
import numpy as np

print("Starting BPB gate check...", flush=True)
model = BILM()
data  = open('data/enwik8','rb').read(200_000)
losses = []
out = {}
t0 = time.time()

for i, b in enumerate(data):
    r = model.observe(b, learn=True)
    losses.append(r.loss_bits)

    if (i+1) in [10000, 50000, 100000, 150000, 200000]:
        bpb   = round(bits_per_byte(losses[-10000:]), 4)
        finite = np.isfinite(losses[-500:]).all()
        elapsed = (time.time()-t0)/60
        out[i+1] = bpb
        print(f"tokens={i+1:>7,}  BPB={bpb}  finite={finite}  elapsed={elapsed:.1f}min", flush=True)

Path('experiments/results').mkdir(parents=True, exist_ok=True)
Path('experiments/results/phase_gates_check.json').write_text(json.dumps(out, indent=2))

gates = json.loads(Path('experiments/results/gates.json').read_text())
bpb_100k = out.get(100000, 999)

print()
print("=" * 50)
print("GATE RESULTS (evaluated at 100K tokens)")
print("=" * 50)
results = {}
for name, target in [
    ('phase_2_dar',   gates['phase_2_dar']),
    ('phase_3_dap',   gates['phase_3_dap']),
    ('phase_4_fales', gates['phase_4_fales']),
    ('phase_5_srs',   gates['phase_5_srs']),
]:
    status = 'PASS' if bpb_100k < target else 'FAIL'
    results[name] = status
    print(f"[{status}] {name}: {bpb_100k} vs target < {target}")

Path('experiments/results/gate_verdicts.json').write_text(json.dumps({
    "bpb_at_100k": bpb_100k,
    "gates": results,
    "all_pass": all(v == "PASS" for v in results.values())
}, indent=2))

print()
overall = "ALL GATES PASSED" if all(v=="PASS" for v in results.values()) else "SOME GATES FAILED"
print(f"VERDICT: {overall}")
