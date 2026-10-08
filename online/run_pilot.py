import sys, json, torch
torch.set_num_threads(4)
import experiment as E
d = E.Data()
bbs = sys.argv[1].split(","); out = sys.argv[2]
rows, hist = E.run_grid(d, bbs, ("scratch", "jepa_ft", "jepa_probe", "recon_ft", "weak_ft"), (5, 20, 100), (0, 1), save=out, log=lambda s: print(s, flush=True))
m, per = E.unet_lastbin(d)
print("UNET last bin", {k: round(v, 3) for k, v in m.items()}, flush=True)
json.dump({"rows": rows, "unet": m}, open(out, "w"))
