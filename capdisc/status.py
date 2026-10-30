#!/usr/bin/env python3
"""Write STATUS.md summarising every phase (standard library only)."""
import glob, json, os, re, time

R = 'results'
out = []
now = time.strftime('%Y-%m-%d %H:%M:%S %Z')
la = os.getloadavg()
out.append(f"# capdisc status — {now}\n")
out.append(f"host load {la[0]:.1f} / {la[1]:.1f} / {la[2]:.1f} (1/5/15 min), {os.cpu_count()} threads\n")

# ---------------------------------------------------------------- ladder
out.append("\n## Floating-point ladder  (floating-point checked, not rigorous: D >= D_best - eta; labels as in VERIFICATION.md)\n")
out.append("| N | eta | status | cells | time | note |\n|---|---|---|---|---|---|\n")
for f in sorted(glob.glob(f'{R}/ladder_N*_eta*.out') + glob.glob(f'{R}/early_N*_eta*.out'), key=lambda s: (int(re.search(r'N(\d+)', s).group(1)), -float(re.search(r'eta([0-9.e-]+)\.out', s).group(1)))):
    N = re.search(r'N(\d+)', f).group(1); eta = re.search(r'eta([0-9.e-]+)\.out', f).group(1)
    txt = open(f).read()
    m = re.search(r'cells=(\d+).*survivors=(\d+).*time=(\d+)s', txt)
    if m:
        st = ('float-checked' if ('CERTIFIED' in txt or 'FLOAT-CHECKED' in txt) else f'survivors {int(m.group(2)):,}')
        out.append(f"| {N} | {eta} | {st} | {int(m.group(1)):,} | {int(m.group(3))/3600:.2f} h | {'early (run_more)' if 'early_' in f else ''} |\n")
    else:
        lg = f[:-4] + '.log'
        last = open(lg).read().strip().split('\n')[-1] if os.path.exists(lg) else ''
        mm = re.search(r'\[\s*(\d+)s\] roots (\d+)/(\d+)\s+cells (\d+)', last)
        prog = f"roots {mm.group(2)}/{mm.group(3)}, {int(mm.group(4)):,} cells, {int(mm.group(1))/3600:.2f} h" if mm else ''
        stale = os.path.exists(lg) and time.time() - os.path.getmtime(lg) > 900
        prog = ('stopped/timed out ' if stale else 'running ') + prog
        out.append(f"| {N} | {eta} | {prog} | | | |\n")

# ---------------------------------------------------------------- search
out.append("\n## Search (best classes)\n")
for f in sorted(glob.glob('logs/search_N*.log'), key=lambda s: int(re.search(r'N(\d+)', s).group(1))):
    N = re.search(r'N(\d+)', f).group(1)
    lines = [l for l in open(f).read().split('\n') if 'round' in l]
    if lines:
        out.append(f"- **N={N}**: {lines[-1].strip()}\n")
    js = f'{R}/search_N{N}.json'
    if os.path.exists(js):
        rows, seen = [], {}
        for r in json.load(open(js)):          # group classes with equal D (flat families)
            k = round(r['D'], 9)
            if k in seen:
                seen[k]['hits'] += r['hits']; seen[k]['n'] += 1
            else:
                seen[k] = dict(r, n=1); rows.append(seen[k])
        out.append("\n  | D | classes | hits | |A| | rank | sharp | min λ | |Aut| |\n  |---|---|---|---|---|---|---|---|\n")
        for r in rows[:6]:
            out.append(f"  | {r['D']:.12f} | {r['n']} | {r['hits']} | {r['active']} | {r['rank']}/{r['dimq']} | {r['sharp']} | {r['min_lambda']:.2e} | {r['aut']} |\n")
        out.append("\n")

# ---------------------------------------------------------------- identification
out.append("\n## Identification\n")
for f in sorted(glob.glob(f'{R}/identify_N*.json')):
    N = re.search(r'N(\d+)', f).group(1)
    for r in json.load(open(f)):
        mp = r.get('minpoly_D')
        deg = f"degree {len(mp)-1}" if mp else ('not found' if r.get('sharp') else 'n/a (not sharp)')
        out.append(f"- N={N}, D={r['D_float']:.12f}: minimal polynomial {deg}\n")
for f in sorted(glob.glob('logs/deep_identify_N*.log')):
    N = re.search(r'N(\d+)', f).group(1)
    out.append(f"- deep N={N}: " + ' | '.join(l.strip() for l in open(f).read().strip().split('\n')[-3:]) + "\n")
for f in sorted(glob.glob(f'{R}/minpoly_N*.json')):
    r = json.load(open(f)); out.append(f"- **N={r['N']}: minimal polynomial of degree {len(r['minpoly_ascending'])-1} found**\n")

# ---------------------------------------------------------------- msolve
out.append("\n## Exact elimination (msolve)\n")
for f in sorted(glob.glob('logs/msolve_*.log')):
    b = f[len('logs/msolve_'):-4]
    o = f'{R}/{b}.msolve.out'
    if os.path.exists(o) and os.path.getsize(o) > 0:
        head = open(o).read(400).replace('\n', ' ')
        res = 'positive-dimensional (needs more saturation)' if head.startswith('[1,') else f'done ({os.path.getsize(o):,} bytes)'
    else:
        res = 'running'
    out.append(f"- {b}: {res}\n")

# ---------------------------------------------------------------- global proof (run_proof.sh)
pr = sorted(glob.glob(f'{R}/proof_N*.out')) + sorted(glob.glob(f'{R}/proof_N*.refine.json'))
if pr or os.path.exists('logs/proof.log'):
    out.append("\n## Global proof (interval-verified mesh + Arb-verified refinement + Krawczyk-verified local checks; second kernels below, see VERIFICATION.md)\n")
    for f in sorted(glob.glob(f'{R}/localcert_*.json')):
        d = json.load(open(f)); out.append(f"- local verification {d['name']}: {'Krawczyk-verified (Arb)' if d.get('verified', d.get('certified')) else 'failed'}, box {d['box_radius']}, D* in {d['images'][0].get('D_enclosure')}\n")
    for f in sorted(glob.glob('certificates/symcert_*.json')):
        d = json.load(open(f)); out.append(f"- symmetric verification {d['name']} image {d.get('image', 0)}: {'Krawczyk-verified (Arb)' if d.get('verified', d.get('certified')) else 'failed'} (group of order {d['group_order']}, D* in {d['D_s']})\n")
    for f in sorted(glob.glob(f'{R}/proof_N*.out')):
        out.append(f"- {os.path.basename(f)}: {open(f).read().strip().splitlines()[0] if open(f).read().strip() else 'running'}\n")
    for f in sorted(glob.glob(f'{R}/proof_N*.refine.json')):
        d = json.load(open(f)); out.append(f"- refine N={d['N']}: {'**Arb-verified, no cell left**' if d.get('verified', d.get('proved')) else 'NOT verified'}: input {d['input_cells']:,}, cells {d['cells']:,}, failed {d['failed']}, {d['time']/3600:.2f} h\n")
    for f in sorted(glob.glob('logs/proof_refine_N*.log')):
        last = open(f).read().strip().split('\n')[-1:] 
        out.append(f"- {os.path.basename(f)}: {last[0] if last else ''}\n")

# ---------------------------------------------------------------- second-kernel pipeline (run_xcheck.sh)
if os.path.exists('logs/xcheck.log'):
    out.append("\n## Second-kernel checks (VERIFICATION.md: cross-verified = two kernels agree)\n")
    for f in sorted(glob.glob('certificates/xcheck_*.json')):
        d = json.load(open(f)); ok = d.get('verified', None)
        out.append(f"- {os.path.basename(f)}: {'agrees' if ok else ('see file' if ok is None else 'DOES NOT AGREE')}\n")
    for f in sorted(glob.glob(f'{R}/xproof_N*.out')):
        out.append(f"- {os.path.basename(f)}: {open(f).read().strip().splitlines()[0] if open(f).read().strip() else 'running'}\n")
    for f in sorted(glob.glob(f'{R}/xproof_N*.xrefine.json')):
        d = json.load(open(f)); out.append(f"- xrefine N={d['N']}: {'**verified (mpmath.iv), no cell left**' if d['verified'] else 'NOT verified'}: input {d['input_cells']:,}, cells {d['cells']:,}, failed {d['failed']}, {d['time']/3600:.2f} h\n")
    out.append("```\n" + '\n'.join(open('logs/xcheck.log').read().strip().split('\n')[-6:]) + "\n```\n")

# ---------------------------------------------------------------- atlas
for f in sorted(glob.glob(f'{R}/atlas_N*.atlas.json')):
    a = json.load(open(f))
    out.append(f"\n## Atlas N={a['N']}: {a['survivors']:,} survivor cells, {len(a['classes'])} local-min classes below threshold\n")
    for c in a['classes'][:8]:
        out.append(f"- D={c['D']:.10f} hits={c['hits']} sharp={c['sharp']} |Aut|={c['aut']}\n")

out.append("\n## Recent orchestrator log\n```\n")
if os.path.exists('logs/run_all.log'):
    out.append('\n'.join(open('logs/run_all.log').read().strip().split('\n')[-12:]) + '\n')
out.append("```\n")
open('STATUS.md', 'w').write(''.join(out))
