"""check/check_chain.py -- does the certificate chain of pipeline A fit together?  (PROOF.md Sections 3, 3a, 6)

Independent of the proof code: uses only the Python standard library (json, hashlib, fractions, subprocess for git),
and does no numerics beyond exact rational comparisons of the values stored in the certificates. It checks that

  C1  the local certificate is verified for every image, is for this N, and lists the images the papers expect; at
      the level "verified" no image has redundant worst caps (extra_caps): without Step 5 nothing shows they are
      <= D* at q* (audit 2026-11-04, finding 2);
  C2  every image has a verified symmetric certificate that refers to it, with the papers' P_N, P_N irreducible
      and identified as the factor that vanishes, and with the symmetric zero inside that image's enclosure;
  C3  the congruence certificate covers every image and is verified;
  C4  the mesh ran in proof mode (rigorous build, every survivor kept, no exclusion balls, no cell cap), for this N,
      with threshold T_mesh >= the refinement threshold, and its survivor files are bound to its stamp by SHA-256
      (route "stamp": hashes in mesh.done; route "sidecar": a stamp written before 2026-11-04 without hashes, and
      mesh.surv.sha256 = "# stamp mesh.done sha256 <hex>  hashed <time>" + sha256sum lines, accepted only if <hex> is
      the SHA-256 of this mesh.done); the files hold at least as many records as the stamp counts survivors.  If
      there is a second pass (mesh2.done), the same for it, it ran in the same directory on the first pass's files
      (by hash: from_stamp = SHA-256 of mesh.done and in.<t> = the first pass's survivor hashes; for a stamp
      without them, by directory name and record count, noted as the weaker route), and it read every record
      of them (input_cells = the records in mesh.surv.*, >= the first pass's survivor count);
  C5  the refinement is verified with no failed cell, for this N, used T = the largest upper end of the D*
      enclosures (level "verified": T above it), excluded exactly the boxes of the local certificate, read the
      local certificate, the last mesh stamp and exactly the last pass's survivor files that are in this directory
      (SHA-256 of each), and read every record of them;
  C6  D* < T strictly, with D* taken from the 128-bit symmetric enclosure (this gives D >= T > D* outside the
      boxes, PROOF.md Section 3);
  C7  N D* lies in the papers' isolating interval of P_N, and D* agrees with the digits printed in the papers;
  C8  all certificates come from commits with the same proof code (proof/, csrc/, verify.sh, build.sh): the Python
      stamps and, for mesh stamps that record it, the commit the mesh binary was built from, whose C sources must
      have the SHA-256 baked into the binary; no stamp reports a modified tree.  The commits, the checker's own
      commit, and whether the checker's proof code is the certificates' are recorded in the output (a difference
      there is flagged, not rejected: an older chain stays checkable by a newer checker).

Flags: caveats that do not make the chain incomplete but limit what it shows about provenance (the certificates'
proof code differs from this checkout; stamps from before 2026-11-04: tree state not evidence, no mesh provenance,
survivors bound by a sidecar, second-pass input bound by count). Printed as FLAG lines above the verdict, counted in
the verdict ("chain complete (2 flags)"), and listed in the result under 'flags'.

Levels: "full" (N = 5, 6; Steps 4-6, formerly called "certified") and "verified" (N = 7: Step 4 only, so C2 and C3
are skipped, C5 requires T > the largest upper end of the local D* enclosures (verify.sh takes the next double),
C6 uses the local enclosures, and C7 compares them with the reported digits only).

usage: python check/check_chain.py N DIR [--out FILE]      (DIR as written by verify.sh, e.g. results/verify_N5)
writes FILE (default: DIR.chain_N<N>.json next to DIR, never into DIR) and exits 0 if every check passes.
"""
import hashlib, json, os, re, subprocess, sys
from fractions import Fraction as Q
from urllib.parse import unquote

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
CODE = ('proof', 'csrc', 'verify.sh', 'build.sh')          # git pathspecs relative to REPO (the -C directory)
LEVEL_ALIASES = {'certified': 'full'}                     # the old name of the level "full" (before 2026-11-04)
PROOF_FLAGS = ('-DRIGOROUS', '-ffp-contract=off', '-fno-fast-math')   # TRUST.md: part of the trusted base


class GitError(RuntimeError):
    pass


def git(*args, strip=True):
    """git in REPO; fails loudly (exit status or anything on stderr)"""
    try:
        r = subprocess.run(['git', '-C', REPO, *args], capture_output=True)
    except OSError as e:
        raise GitError(f'cannot run git: {e}') from e
    err = r.stderr.decode(errors='replace').strip()
    if r.returncode != 0 or err:
        raise GitError(f"git {' '.join(args)}: exit {r.returncode}: {err or '(no message)'}")
    return r.stdout


def same_code(a, b):
    """(same, files): do commits a and b have the same proof code (CODE)? files: the paths that differ"""
    out = git('diff', '--name-only', a, b, '--', *CODE).decode()
    files = [f for f in out.splitlines() if f.strip()]
    return not files, files


def tree_dirty():
    """files under CODE and check/ that differ from HEAD in this checkout (untracked included)"""
    out = git('status', '--porcelain', '--untracked-files=all', '--', *CODE, 'check').decode()
    return [ln[3:] for ln in out.splitlines() if ln.strip()]


def src_sha_at(commit, files):
    """SHA-256 of the C sources at `commit`, computed as build.sh does: sha256 of the sha256sum lines of csrc/<f>"""
    lines = ''.join(f"{hashlib.sha256(git('show', f'{commit}:./csrc/{f}', strip=False)).hexdigest()}  csrc/{f}\n" for f in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def hexq(s):                      # a hex float string or a JSON number, as an exact rational
    return Q(float.fromhex(s)) if isinstance(s, str) else Q(s)


def ball(s):                      # an Arb ball string '[m +/- r]' (or an exact number) -> (lo, hi), exact rationals
    s = s.strip()
    if s.startswith('['):
        m, r = s[1:-1].split('+/-')
        m, r = Q(m.strip()), Q(r.strip())
        return m - r, m + r
    return Q(s), Q(s)


def parse_stamp_text(text):
    """a capcert completion stamp: whitespace-separated key=value tokens; %-encoded values decoded"""
    out = {}
    for tok in text.split():
        if '=' not in tok: raise ValueError(f'stamp token without "=": {tok!r}')
        k, v = tok.split('=', 1)
        out[k] = unquote(v)
    return out


def public_dir(d):
    r = os.path.relpath(os.path.abspath(d), REPO)
    return r if not r.startswith('..') else '.../' + os.path.basename(os.path.abspath(d).rstrip('/'))


class Incomplete(Exception):
    pass


class Checker:
    def __init__(self, N, DIR):
        self.N, self.DIR = N, DIR
        self.checks, self.notes, self.routes, self.code, self.flags = {}, [], {}, {}, []
        self._sha = {}

    # ---------------------------------------------------------------- helpers
    def check(self, key, ok, why=''):
        self.checks[key] = bool(ok) and self.checks.get(key, True)
        if not ok: self.notes.append(f'{key}: {why}')
        return bool(ok)

    def flag(self, kind, msg):
        """a caveat that does not make the chain incomplete but limits what it shows about provenance: printed
        as FLAG above the verdict and listed in the result under 'flags'"""
        self.flags.append(dict(kind=kind, message=msg))

    def path(self, fn):
        return os.path.join(self.DIR, fn)

    def fsha(self, fn):
        if fn not in self._sha: self._sha[fn] = sha(self.path(fn))
        return self._sha[fn]

    def load(self, fn):
        if not os.path.exists(self.path(fn)):
            raise Incomplete(f'missing certificate: {fn}')
        with open(self.path(fn)) as f: return json.load(f)

    def load_stamp(self, fn):
        if not os.path.exists(self.path(fn)):
            raise Incomplete(f'missing certificate: {fn}')
        with open(self.path(fn)) as f: return parse_stamp_text(f.read())

    def surv_files(self, P, rec, key):
        """the survivor files P.surv.<t> in DIR: contiguous from 0, whole records. -> (names, total records)"""
        idx = sorted(int(m.group(1)) for m in (re.fullmatch(re.escape(P) + r'\.surv\.(\d+)', f) for f in os.listdir(self.DIR)) if m)
        names = [f'{P}.surv.{t}' for t in idx]
        self.check(key, idx == list(range(len(idx))), f'{P}.surv.*: file numbers {idx} are not 0..{len(idx) - 1}')
        total = 0
        for fn in names:
            size = os.path.getsize(self.path(fn))
            self.check(key, size % rec == 0, f'{fn}: partial record')
            total += size // rec
        return names, total

    def bind_survivors(self, P, stamp, rec, key):
        """bind the survivor files of pass P to its stamp by SHA-256 (route 'stamp' or 'sidecar').
        -> (hashes {name: hex}, total records)"""
        names, total = self.surv_files(P, rec, key)
        if 'files' in stamp:
            want = {f'{P}.surv.{t}': stamp.get(f'surv.{t}') for t in range(int(stamp['files']))}
            self.check(key, names == list(want),
                       f'{P}.surv.*: files {names}, the stamp lists {list(want)}')
            for fn in names:
                self.check(key, want.get(fn) == self.fsha(fn), f'{fn}: SHA-256 differs from {P}.done')
            if 'records' in stamp: self.check(key, int(stamp['records']) == total, f'{P}: {total} records, the stamp says {stamp["records"]}')
            self.routes[P] = dict(binding='stamp', files=len(names), records=total)
            return want, total
        side = self.path(f'{P}.surv.sha256')
        if not os.path.exists(side):
            self.check(key, False, f'{P}.done has no survivor hashes (written before 2026-11-04) and there is no {P}.surv.sha256')
            self.routes[P] = dict(binding='none'); return {}, total
        with open(side) as f: lines = [ln.rstrip('\n') for ln in f if ln.strip()]
        m = re.fullmatch(r'#\s*stamp\s+(\S+)\s+sha256\s+([0-9a-f]{64})\s+hashed\s+(.*)', lines[0].strip()) if lines else None
        if not self.check(key, m is not None, f'{P}.surv.sha256: first line is not "# stamp {P}.done sha256 <hex>  hashed <time>"'):
            self.routes[P] = dict(binding='none'); return {}, total
        self.check(key, os.path.basename(m.group(1)) == f'{P}.done', f'{P}.surv.sha256 names the stamp {m.group(1)}')
        self.check(key, m.group(2) == self.fsha(f'{P}.done'), f'{P}.surv.sha256: the stamp hash is not the SHA-256 of {P}.done')
        want = {}
        for ln in lines[1:]:
            mm = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', ln.strip())
            if not self.check(key, mm is not None, f'{P}.surv.sha256: bad line {ln!r}'): continue
            bn = os.path.basename(mm.group(2))
            self.check(key, re.fullmatch(re.escape(P) + r'\.surv\.\d+', bn) is not None and bn not in want, f'{P}.surv.sha256: unexpected or repeated entry {bn}')
            want[bn] = mm.group(1)
        self.check(key, sorted(want) == sorted(names), f'{P}.surv.sha256 lists {sorted(want)}, the directory has {sorted(names)}')
        for fn in names:
            self.check(key, want.get(fn) == self.fsha(fn), f'{fn}: SHA-256 differs from {P}.surv.sha256')
        self.routes[P] = dict(binding='sidecar', sidecar=f'{P}.surv.sha256', stamp_sha256=m.group(2), hashed=m.group(3).strip(),
                              files=len(names), records=total,
                              note='stamp written before 2026-11-04 (no hashes); survivor files bound by the sidecar, '
                                   'which names the SHA-256 of the stamp')
        self.flag('survivors_bound_by_sidecar', f'{P}.surv.* are bound by {P}.surv.sha256 (hashed {m.group(3).strip()}), '
                                                f'not by hashes in {P}.done (stamp written before 2026-11-04)')
        return want, total

    def proof_mode(self, s, P, key):
        self.check(key, s.get('mode') == 'rigorous' and float.fromhex(s.get('keep', 'nan')) == 1.0 and int(s.get('balls', -1)) == 0
                   and int(s.get('N', -1)) == self.N, f'{P} not run in proof mode for N = {self.N}')
        if 'maxc' in s: self.check(key, int(s['maxc']) == 0, f'{P}: --max-cells-per-root was used')
        if 'build' in s:
            self.check(key, s['build'] == 'rigorous', f'{P}: written by the {s["build"]} build')
            flags = s.get('flags', '').split()
            self.check(key, all(f in flags for f in PROOF_FLAGS), f'{P}: compiler flags {flags} lack one of {PROOF_FLAGS}')

    # ---------------------------------------------------------------- the checks
    def run(self, paper):
        N = self.N; name = paper['name']
        level = paper.get('level', 'full'); level = LEVEL_ALIASES.get(level, level)
        VERIFIED_ONLY = level == 'verified'
        self.level = level
        rec = (2 * (2 * N - 3) + 1) * 8

        lc = self.load(f'localcert_{name}.json')
        imgs = lc['images']
        self.check('C1', lc['verified'] and all(i['verified'] for i in imgs), 'local certificate not verified')
        self.check('C1', len(imgs) == paper['images'], f"{len(imgs)} images, papers expect {paper['images']}")
        self.check('C1', lc.get('N', N) == N and lc.get('name', name) == name, f"local certificate is for {lc.get('name')}, N = {lc.get('N')}")
        if VERIFIED_ONLY:
            bad = [i['image'] for i in imgs if i.get('extra_caps')]
            self.check('C1', not bad, f'images {bad} have redundant worst caps (extra_caps), which the level "verified" cannot close')

        def local_enc(i):
            return tuple(hexq(x) for x in i['D_enclosure_hex']) if 'D_enclosure_hex' in i else tuple(Q(x) for x in i['D_enclosure'])

        Dstar, cg, scs = [], None, []
        if VERIFIED_ONLY:
            Dstar = [local_enc(i) for i in imgs]
            self.notes.append('level verified: no symmetric (C2) or congruence (C3) step')
        else:
            for i in imgs:
                k = i['image']
                sc = self.load(f'symcert_{name}_img{k}.json'); scs.append(sc)
                self.check('C2', sc['verified'] and sc['image'] == k, f'image {k}: symmetric certificate not verified')
                self.check('C2', sc.get('N', N) == N and sc.get('name', name) == name, f'image {k}: symmetric certificate for another N')
                self.check('C2', sc.get('P_N') == paper['P_N'], f'image {k}: P_N differs from the papers')
                self.check('C2', sc.get('P_N_irreducible') is True and sc.get('P_N_is_the_factor') is True, f'image {k}: P_N not shown irreducible / not the factor')
                self.check('C2', sc.get('group_order') == paper['group_order'], f'image {k}: group order')
                dlo, dhi = ball(sc['s_enclosure'][-1])
                elo, ehi = local_enc(i)
                self.check('C2', elo <= dlo and dhi <= ehi, f'image {k}: symmetric D* not inside the local enclosure')
                Dstar.append((dlo, dhi))
            cg = self.load(f'congruence_{name}.json')
            self.check('C3', cg['congruent'] and cg.get('name', name) == name
                       and sorted(o['image'] for o in cg['images']) == sorted(i['image'] for i in imgs)
                       and all(o['contained_in_Z0'] and o['verified_image'] for o in cg['images']), 'congruence')

        # ------------------------------------------------ C4: the mesh pass(es) and their survivor files
        s1 = self.load_stamp('mesh.done')
        self.proof_mode(s1, 'mesh', 'C4')
        h1, rec1 = self.bind_survivors('mesh', s1, rec, 'C4')
        self.check('C4', rec1 >= int(s1['survivors']), f"mesh: {rec1} records, the stamp counts {s1['survivors']} survivors")
        T_mesh = [Q(float.fromhex(s1['T']))]
        TWO = os.path.exists(self.path('mesh2.done'))
        last, lastP, lasth, lastrec = s1, 'mesh', h1, rec1
        if TWO:
            s2 = self.load_stamp('mesh2.done')
            self.proof_mode(s2, 'mesh2', 'C4')
            T_mesh.append(Q(float.fromhex(s2['T'])))
            frm = s2.get('from', '')
            self.check('C4', os.path.basename(frm) == 'mesh', f'second pass read {frm!r}, not the first pass "mesh"')
            r2 = self.routes.setdefault('mesh2_input', {})
            if 'same_dir' in s2:
                self.check('C4', s2['same_dir'] == '1' and s2.get('prefix') == 'mesh2', 'second pass: input and output not in one directory')
                r2['directory'] = 'same_dir=1 in mesh2.done'
            else:      # stamp before 2026-11-04: from= is the path as given (relative to capdisc/ in verify.sh)
                d_from = os.path.dirname(frm)
                same = os.path.realpath(os.path.join(REPO, d_from)) == os.path.realpath(self.DIR)
                same_name = os.path.basename(d_from.rstrip('/')) == os.path.basename(os.path.realpath(self.DIR))
                self.check('C4', same or same_name, f'second pass read {frm!r}, which is not in this directory')
                r2['directory'] = 'from= resolves to this directory' if same else 'from= names a directory of the same name (directory moved)'
            if 'from_stamp' in s2:
                self.check('C4', s2['from_stamp'] == self.fsha('mesh.done'), 'second pass read a different first-pass stamp')
                ins = {f'mesh.surv.{t}': s2.get(f'in.{t}') for t in range(int(s2.get('input_files', -1)))}
                self.check('C4', ins == h1, 'second pass read survivor files other than mesh.surv.* (SHA-256)')
                r2['binding'] = 'hashes (from_stamp, in.<t>)'
            else:
                r2['binding'] = 'record count only (stamp written before 2026-11-04)'
                self.flag('second_pass_input_by_count', 'mesh2.done predates input hashes: the second pass is bound to '
                                                        'mesh.surv.* only by the record count and the directory name')
            self.check('C4', int(s2['input_cells']) == rec1, f"second pass read {s2['input_cells']} cells, mesh.surv.* hold {rec1}")
            self.check('C4', int(s2['input_cells']) >= int(s1['survivors']), f"second pass read {s2['input_cells']} cells, the first pass counted {s1['survivors']}")
            h2, rec2 = self.bind_survivors('mesh2', s2, rec, 'C4')
            self.check('C4', rec2 >= int(s2['survivors']), f"mesh2: {rec2} records, the stamp counts {s2['survivors']} survivors")
            last, lastP, lasth, lastrec = s2, 'mesh2', h2, rec2

        # ------------------------------------------------ C5: the refinement
        rf = self.load(f'{lastP}.refine.json')
        T = max(local_enc(i)[1] for i in imgs)
        self.check('C4', all(t >= T for t in T_mesh), 'mesh threshold below the refinement threshold')
        self.check('C5', rf['verified'] and rf['failed'] == 0, 'refinement left cells')
        self.check('C5', rf.get('N', N) == N, f"refinement is for N = {rf.get('N')}")
        if VERIFIED_ONLY:
            self.check('C5', hexq(rf['T_hex']) > T, 'refinement threshold is not above the largest upper end of the D* enclosures')
            self.check('C4', all(t >= hexq(rf['T_hex']) for t in T_mesh), 'mesh threshold below the refinement threshold')
            T = hexq(rf['T_hex'])
        else:
            self.check('C5', hexq(rf['T_hex']) == T, 'refinement threshold is not the largest upper end of the D* enclosures')
        boxes = sorted((tuple(hexq(x) for x in b['center']), hexq(b['radius'])) for b in rf['exclusion_boxes'])
        local_boxes = sorted((tuple(Q(x) for x in i['q_center']), Q(lc['box_radius'])) for i in imgs)
        self.check('C5', boxes == local_boxes, 'exclusion boxes differ from the local boxes')
        # inputs are recorded by path relative to the repository at run time; match them by file name, so that the
        # directory can be moved (e.g. from results/verify_N5 to certificates/verify_N5)
        ins = {}
        for p, h in rf['stamp']['inputs'].items():
            bn = os.path.basename(p)
            self.check('C5', bn not in ins, f'refinement records two inputs named {bn}')
            ins[bn] = h
        for fn, what in ((f'localcert_{name}.json', 'local certificate'), (f'{lastP}.done', 'mesh stamp')):
            self.check('C5', ins.get(fn) == self.fsha(fn), f'refinement read a different {what}')
        surv_in = {fn: h for fn, h in ins.items() if '.surv.' in fn}
        self.check('C5', sorted(surv_in) == sorted(lasth), f'refinement read {sorted(surv_in)}, the last pass wrote {sorted(lasth)}')
        for fn in sorted(set(surv_in) & set(lasth)):
            self.check('C5', os.path.exists(self.path(fn)) and surv_in[fn] == self.fsha(fn), f'{fn}: the refinement read other content (SHA-256)')
        self.check('C5', rf['input_cells'] == lastrec, f"refinement read {rf['input_cells']} cells, {lastP}.surv.* hold {lastrec}")
        self.check('C5', rf['input_cells'] >= int(last['survivors']), 'refinement read fewer survivors than the mesh wrote')

        self.check('C6', all(dhi < T for _, dhi in Dstar), 'D* is not strictly below T')

        pr = Q(paper['D_printed']); ulp = Q(1, 10 ** (len(paper['D_printed']) - 2))
        if not VERIFIED_ONLY:
            lo, hi = (Q(x) for x in paper['u_interval'])
            self.check('C7', all(lo <= N * dlo and N * dhi <= hi for dlo, dhi in Dstar), 'N D* outside the isolating interval')
        self.check('C7', all(pr <= dlo and dhi < pr + ulp for dlo, dhi in Dstar), 'D* does not match the printed digits')

        # ------------------------------------------------ C8: one proof code, clean trees, the mesh binary's sources
        named = [('local', lc['stamp']), ('refinement', rf['stamp'])]
        if not VERIFIED_ONLY:
            named += [('congruence', cg['stamp'])] + [(f'symmetric img{sc["image"]}', sc['stamp']) for sc in scs]
        for what, s in named:
            self.check('C8', bool(s.get('commit')), f'{what}: no commit recorded')
            self.check('C8', s.get('tree_dirty') is False, f'{what}: modified proof code ({s.get("tree_dirty_files", "")})')
        legacy = [what for what, s in named if 'tree_checked' not in s]
        if legacy:
            self.flag('tree_state_unreliable', f'the stamps of {legacy} predate the fixed tree check (2026-11-04): their '
                                               'tree_dirty=false is not evidence that the proof code was unmodified')
        commits = {what: s.get('commit') for what, s in named}
        mesh_commits = {}
        for P, s in (('mesh', s1),) + ((('mesh2', last),) if TWO else ()):
            if 'commit' not in s:
                mesh_commits[P] = None
                self.flag('mesh_provenance_missing', f'{P}.done predates provenance (2026-11-04): the commit, sources '
                                                     'and flags of its binary are not recorded')
                continue
            mesh_commits[P] = s['commit']
            self.check('C8', re.fullmatch(r'[0-9a-f]{40}', s['commit']) is not None, f'{P}: binary built without a commit ({s["commit"]})')
            self.check('C8', s.get('tree_dirty') == '0', f'{P}: binary built from a modified tree (tree_dirty={s.get("tree_dirty")})')
            if re.fullmatch(r'[0-9a-f]{40}', s['commit']):
                try:
                    want = src_sha_at(s['commit'], s['src'].split(','))
                    self.check('C8', want == s.get('src_sha256'), f'{P}: the C sources of {s["commit"][:8]} do not have the hash baked into the binary')
                except GitError as e:
                    self.check('C8', False, f'{P}: {e} (a shallow clone lacks the commits)')
        allc = sorted({c for c in list(commits.values()) + list(mesh_commits.values()) if c})
        try:
            for c in allc[1:]:
                same, files = same_code(allc[0], c)
                self.check('C8', same, f'commits {allc[0][:8]} and {c[:8]} differ in the proof code: {files}')
        except GitError as e:
            self.check('C8', False, f'{e} (a shallow clone lacks the commits)')
        self.check('C8', bool(allc), 'no commit recorded')
        self.code = dict(certificate_commits=commits, mesh_commits=mesh_commits)
        try:
            me = git('rev-parse', 'HEAD').decode().strip(); dirty = tree_dirty()
            self.code.update(checker_commit=me, checker_tree_dirty=bool(dirty))
            if dirty: self.code['checker_dirty_files'] = dirty
            if allc:
                same, files = same_code(allc[0], me)
                self.code.update(proof_code_same_as_checker=same, differing_files=files)
                if not same:
                    self.flag('proof_code_differs_from_checker', f'the certificates come from the proof code of '
                              f'{allc[0][:8]}, which differs from this checkout ({me[:8]}) in {files}: audit the code of '
                              f'{allc[0][:8]}, not of this checkout')
        except GitError as e:
            self.code.update(checker_commit=None, checker_error=str(e))
            self.flag('checker_commit_unknown', f'cannot compare with the checker commit: {e}')
        self.T, self.Dstar = T, Dstar


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    out = None
    if '--out' in argv:
        out = argv[argv.index('--out') + 1]; args.remove(out)
    if len(args) != 2: sys.exit('usage: python check/check_chain.py N DIR [--out FILE]')
    N, DIR = int(args[0]), args[1]
    if '--hash-survivors' in argv: print('note: --hash-survivors is no longer needed: survivor files are always hashed')
    with open(os.path.join(HERE, 'paper_values.json')) as f: paper = json.load(f)[str(N)]
    out = out or os.path.abspath(DIR).rstrip('/') + f'.chain_N{N}.json'
    ch = Checker(N, DIR)
    try:
        ch.run(paper)
    except Incomplete as e:
        print(f'{e}\nCHAIN INCOMPLETE'); return 1
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as e:
        print(f'malformed certificate: {type(e).__name__}: {e}\nCHAIN INCOMPLETE'); return 1
    required = {'C1', 'C4', 'C5', 'C6', 'C7', 'C8'} | (set() if ch.level == 'verified' else {'C2', 'C3'})
    ok = all(ch.checks.values()) and required <= set(ch.checks)
    files = {f: ch.fsha(f) for f in sorted(os.listdir(DIR))
             if (f.endswith('.json') and not f.startswith('chain_')) or f.endswith('.done') or '.surv.' in f}
    res = dict(N=N, name=paper['name'], level=ch.level, directory=public_dir(DIR), checks=ch.checks, notes=ch.notes,
               verdict=('chain complete' + (f' ({len(ch.flags)} flags)' if ch.flags else '')) if ok else 'CHAIN INCOMPLETE',
               flags=ch.flags, routes=ch.routes, code=ch.code,
               T=str(ch.T), D_star_upper=str(max(d for _, d in ch.Dstar)), files=files)
    tmp = out + '.tmp'
    with open(tmp, 'w') as f: json.dump(res, f, indent=1); f.write('\n')
    os.replace(tmp, out)
    for k in sorted(ch.checks): print(f'{k}: {"ok" if ch.checks[k] else "FAIL"}')
    for n_ in ch.notes: print('  ' + n_)
    for P, r in sorted(ch.routes.items()): print(f'  route {P}: {r.get("binding")}' + (f' ({r["directory"]})' if 'directory' in r else ''))
    print(f'  result written to {public_dir(out)}')
    for fl in ch.flags: print(f'FLAG [{fl["kind"]}]: {fl["message"]}')
    print(res['verdict'])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
