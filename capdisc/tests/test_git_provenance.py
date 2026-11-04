"""The tree state in the certificate stamps and the dirty-tree guard of verify.sh (audit 2026-11-04, finding 1).

Before the fix the pathspecs were 'capdisc/proof' etc. relative to capdisc/ itself, so they matched nothing and every
stamp said tree_dirty=False. The synthetic repository below has the same layout (the git top level one directory
above the code), so the test fails with the old pathspecs."""
import os, shutil, subprocess, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from proof import certio                              # noqa: E402


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    top = tmp_path / 'top'; code = top / 'capdisc'
    for f in ('proof/arith.py', 'proof/hints/N5_best.json', 'csrc/capcert.c', 'verify.sh', 'build.sh', 'README.md', 'jobs/x.py'):
        p = code / f; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(f'{f}\n')
    sh('git', 'init', '-q', cwd=top); sh('git', 'add', '-A', cwd=top)
    sh('git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', 'commit', '-qm', 'init', cwd=top)
    return str(code)


def test_clean_tree(repo):
    commit, dirty, files = certio.tree_state(repo)
    assert len(commit) == 40 and not dirty and files == []


@pytest.mark.parametrize('change', ['modify proof/arith.py', 'modify csrc/capcert.c', 'modify verify.sh', 'modify build.sh',
                                    'add proof/new.py', 'add csrc/extra.h', 'delete proof/hints/N5_best.json'])
def test_a_change_to_the_proof_code_makes_the_tree_dirty(repo, change):
    op, f = change.split(); p = os.path.join(repo, f)
    if op == 'delete': os.remove(p)
    else:
        with open(p, 'a') as fh: fh.write('x\n')
    commit, dirty, files = certio.tree_state(repo)
    assert dirty and any(f in x for x in files), files


@pytest.mark.parametrize('f', ['README.md', 'jobs/x.py', 'results/new.json'])
def test_changes_outside_the_proof_code_do_not(repo, f):
    p = os.path.join(repo, f); os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a') as fh: fh.write('x\n')
    assert not certio.tree_state(repo)[1]


def test_stamp_reports_a_modified_proof_file(repo, monkeypatch):
    """the audit's reproduction (i): with proof/arith.py modified, certio.stamp() must say tree_dirty=True"""
    with open(os.path.join(repo, 'proof', 'arith.py'), 'a') as fh: fh.write('# modified\n')
    monkeypatch.setattr(certio, 'REPO', repo)
    s = certio.stamp('test')
    assert s['tree_dirty'] is True and s['tree_checked'] == list(certio.CODE_PATHS)
    assert any('proof/arith.py' in f for f in s['tree_dirty_files'])


def test_git_fails_loudly(tmp_path):
    with pytest.raises(certio.GitError):
        certio.tree_state(str(tmp_path))                 # not a repository
    with pytest.raises(certio.GitError):
        certio._git('rev-parse', '--verify', 'no-such-ref^{commit}')


def test_input_keys_outside_the_repository_are_shortened(tmp_path):
    p = tmp_path / 'some' / 'dir' / 'mesh.done'; p.parent.mkdir(parents=True); p.write_text('x\n')
    s = certio.stamp('test', inputs=[str(p), os.path.join(ROOT, 'proof', 'hints', 'N5_best.json')])
    assert set(s['inputs']) == {'.../mesh.done', 'proof/hints/N5_best.json'}


def test_verify_sh_aborts_on_a_dirty_tree(tmp_path):
    """verify.sh in proof mode refuses to run when the proof code differs from the commit (in a clone of HEAD)"""
    top = sh('git', 'rev-parse', '--show-toplevel', cwd=ROOT).strip()
    clone = tmp_path / 'clone'
    sh('git', 'clone', '-q', '--local', '--no-hardlinks', top, str(clone), cwd=str(tmp_path))
    code = clone / os.path.relpath(ROOT, top)
    with open(code / 'proof' / 'arith.py', 'a') as fh: fh.write('# modified\n')
    r = subprocess.run(['bash', 'verify.sh', '5'], cwd=str(code), capture_output=True, text=True, env=dict(os.environ, PY=sys.executable))
    assert r.returncode == 1 and 'differs from commit' in r.stdout, r.stdout + r.stderr
    assert not (code / 'results' / 'verify_N5').exists()
