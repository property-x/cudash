"""Plots benches/<name>.csv as benches/<name>.svg: GB/s above, time below. Standard library only.

    python3 benches/plot_sweep.py [name]     # name defaults to transfer_sweep

Columns: path,bytes,label,ns and, optionally, p10,p90 (drawn as a band).
"""
import csv, math, os, sys
from collections import defaultdict

name = sys.argv[1] if len(sys.argv) > 1 else "transfer_sweep"
here = os.path.dirname(os.path.abspath(__file__))
lines = open(os.path.join(here, name + ".csv")).read().splitlines()
device = next((l[9:].strip() for l in lines if l.startswith("# device:")), "")
series = defaultdict(list)  # path -> [row]
for r in csv.DictReader(l for l in lines if not l.startswith("#")):
    series[r["path"]].append(r)

palette = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
W, PH, L, R, T = 900, 330, 70, 20, 60
H = T + 2 * PH + 120
xs = [int(r["bytes"]) for s in series.values() for r in s]
x0, x1 = math.log2(min(xs)), math.log2(max(xs))
px = lambda b: L + (math.log2(b) - x0) / (x1 - x0) * (W - L - R)

o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{L}" y="24" font-size="16" font-weight="bold">{name}: host to device copy (copy + synchronize)</text>',
     f'<text x="{L}" y="42" fill="#555">{device}. Line = median, band = 10th to 90th percentile.</text>']
ticks = [b for b in sorted(set(xs)) if b & (b - 1) == 0 and math.log2(b) % 2 == 0]

def panel(top, title, value, ylog, fmt):
    """value(row, key) gives the plotted number for column `key` of a row."""
    keys = ["ns", "p90", "p10"]
    vals = [value(r, k) for s in series.values() for r in s for k in keys if k in r]
    lo, hi = min(vals), max(vals) * 1.15
    if ylog:
        lo = 10 ** math.floor(math.log10(lo)); hi = 10 ** math.ceil(math.log10(hi))
        py = lambda v: top + PH - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * PH
        grid = [10 ** e for e in range(round(math.log10(lo)), round(math.log10(hi)) + 1)]
    else:
        py = lambda v: top + PH - v / hi * PH
        mag = 10 ** math.floor(math.log10(hi / 6))
        step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= hi / 6)
        grid = [i * step for i in range(int(hi / step) + 1)]
    o.append(f'<text x="16" y="{top+PH/2:.0f}" transform="rotate(-90 16 {top+PH/2:.0f})" text-anchor="middle" fill="#333">{title}</text>')
    for g in grid:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{py(g):.1f}" y2="{py(g):.1f}" stroke="#ddd"/>'
                 f'<text x="{L-8}" y="{py(g)+4:.1f}" text-anchor="end">{fmt(g)}</text>')
    for b in ticks:
        lab = f"{b>>10} KiB" if b < 1 << 20 else f"{b>>20} MiB"
        o.append(f'<line x1="{px(b):.1f}" x2="{px(b):.1f}" y1="{top}" y2="{top+PH}" stroke="#eee"/>'
                 f'<text x="{px(b):.1f}" y="{top+PH+16}" text-anchor="middle">{lab}</text>')
    for i, (p, rows) in enumerate(series.items()):
        c = palette[i % len(palette)]
        pts = [(int(r["bytes"]), r) for r in rows]
        if "p10" in rows[0]:
            up = [f"{px(b):.1f},{py(value(r,'p90')):.1f}" for b, r in pts]
            dn = [f"{px(b):.1f},{py(value(r,'p10')):.1f}" for b, r in reversed(pts)]
            o.append(f'<polygon points="{" ".join(up+dn)}" fill="{c}" opacity="0.18"/>')
        d = " ".join(f"{px(b):.1f},{py(value(r,'ns')):.1f}" for b, r in pts)
        o.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        o.extend([f'<circle cx="{px(b):.1f}" cy="{py(value(r,"ns")):.1f}" r="3" fill="{c}"/>' for b, r in pts])

# GB/s: a high time is a low rate, so the percentile band swaps ends.
gbs = lambda r, k: int(r["bytes"]) / float(r[{"p90": "p10", "p10": "p90"}.get(k, k)])
panel(T, "GB/s", gbs, False, lambda g: f"{g:g}")
micros = lambda r, k: float(r[k]) / 1000
panel(T + PH + 40, "time per copy, µs", micros, False, lambda g: f"{g:g}")
y = T + 2 * PH + 82
o.append(f'<text x="{(L+W-R)/2}" y="{y-4}" text-anchor="middle" fill="#555">copy size (log scale); right edge is 2560x2160 f32 = 22.1 MB</text>')
for i, p in enumerate(series):
    lx = L + 10 + i * 170
    o.append(f'<rect x="{lx}" y="{H-24}" width="12" height="12" fill="{palette[i % len(palette)]}"/><text x="{lx+18}" y="{H-14}">{p}</text>')
o.append("</svg>")
open(os.path.join(here, name + ".svg"), "w").write("\n".join(o))
print(f"wrote benches/{name}.svg")
