from bilm import BILM
from bilm.metrics import bits_per_byte
import numpy as np

model = BILM()
data  = open('data/enwik8','rb').read(15_000)
losses = []

for i, b in enumerate(data):
    r = model.observe(b, learn=True)
    losses.append(r.loss_bits)
    if (i+1) in [1000, 5000, 10000, 15000]:
        bpb = round(bits_per_byte(losses[-1000:]), 4)
        finite = np.isfinite(losses[-100:]).all()
        print(f'tokens={i+1:>6,}  BPB={bpb}  finite={finite}', flush=True)
