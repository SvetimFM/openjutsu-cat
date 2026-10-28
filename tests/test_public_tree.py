"""The repo is published as-is: nothing in it may point at the author's own
machines, accounts or working notes, and the README's facts must match the
files they describe."""
import hashlib
import json
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRIVATE = [
    (re.compile(r"\b[A-Z]:\\Users\\", re.I), "personal Windows path"),
    # any drive except system folders and the placeholder roots the docs use
    (re.compile(r"\b[A-Z]:[\\/](?!/|Windows\\|Program Files|takes\\|comfy-models\b|models\b)"), "author's drive letter"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}"), "token"),
    (re.compile(r"[\w.+-]+@[\w-]+\.(com|net|org|io|ai)\b"), "email address"),
    (re.compile(r"PREP-NOTES|DECISION-NEEDED|do not publish", re.I), "internal working note"),
]
# sha256 of lower-cased words that must never appear: the author's machine,
# account and family names. Hashed so that this file doesn't publish them itself.
PRIVATE_WORDS = {
    "83de5bf590789716a4c4bdabc214c369c0ff540250fa4ed7e404cdb97f9aea8f",
    "06b84ee4064869e739f39a7acafd622048966f62f8e889c7486eaf1102dee3fc",
    "626d75cfcb38abfb393bcf504805f7f611ed1ed678bd6d99a04b501acaf04e15",
    "ef8f89441c21d4187cb4df9ead52f1b64c9aa0a67e7ed784933fe1b936c167cd",
    "eea8ec0d450873055807a6a6fd5d701a910b3fa338b741feab1be5320d2bd006",
    "ec47860233d36cad00db3ed18c8ed7d57690cb8689caa86cce93ef75d767e7f1",
    "f8657bc65f1ff0108eb3746eecb1a58bbe99a312df572f4a6118f142b55c0cbb",
    "c7a675e951ff4b13b4196a468152ae6d633ddaa9a500ff3b482c46dc27e0a85b",
}
WORD = re.compile(r"[a-z0-9]+")
BINARY = (".mp4", ".png", ".jpg", ".gif", ".webp")
SELF = os.path.relpath(__file__, ROOT).replace(os.sep, "/")
NOREPLY = "@users.noreply.github.com"


def private_hits(line):
    hits = [what for rx, what in PRIVATE if rx.search(line)]
    if any(hashlib.sha256(w.encode()).hexdigest() in PRIVATE_WORDS for w in WORD.findall(line.lower())):
        hits.append("author's machine, account or family name")
    return hits


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [f for f in out.splitlines() if os.path.isfile(os.path.join(ROOT, f))]
    except (OSError, subprocess.CalledProcessError):
        files = [os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT)
                 if ".git" not in d.split(os.sep) and "__pycache__" not in d for f in fs]
    return sorted(f.replace(os.sep, "/") for f in files)


class PublicTree(unittest.TestCase):
    def test_no_private_content(self):
        hits = []
        for rel in tracked_files():
            if rel.endswith(BINARY) or rel == SELF:
                continue
            with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
                for n, line in enumerate(f, 1):
                    hits += [f"{rel}:{n}: {what}: {line.strip()[:120]}" for what in private_hits(line)]
        self.assertEqual(hits, [], "\n".join(hits))

    def test_no_model_weights_or_outputs_tracked(self):
        bad = [f for f in tracked_files() if f.endswith((".gguf", ".safetensors", ".ckpt", ".bin"))
               or f.startswith(("out/", "output/"))]
        self.assertEqual(bad, [])

    def test_licence_present(self):
        with open(os.path.join(ROOT, "LICENSE"), encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("MIT License"))


def readme():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        return f.read()


class PublicHistory(unittest.TestCase):
    """Pushing publishes every commit, not just the tree: no earlier version of any
    file, no commit message and no author identity may carry private content."""

    def setUp(self):
        try:
            self.commits = git("rev-list", "HEAD").decode().split()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")

    def test_no_private_content_in_any_commit(self):
        hits = []
        seen = set()
        for line in git("rev-list", "--objects", "HEAD").decode().splitlines():
            sha, _, rel = line.partition(" ")
            if not rel or sha in seen or rel.endswith(BINARY) or rel == SELF:
                continue
            seen.add(sha)
            if git("cat-file", "-t", sha).strip() != b"blob":
                continue
            text = git("cat-file", "blob", sha).decode("utf-8", "replace")
            for n, l in enumerate(text.splitlines(), 1):
                hits += [f"{rel}@{sha[:7]}:{n}: {what}: {l.strip()[:100]}" for what in private_hits(l)]
        self.assertEqual(hits, [], "\n".join(hits))

    def test_commit_identities_and_messages(self):
        hits = []
        for c in self.commits:
            ident, msg = git("show", "-s", "--format=%an <%ae> %cn <%ce>%x00%B", c).decode().split("\0", 1)
            emails = re.findall(r"<([^>]*)>", ident)
            hits += [f"{c[:7]}: personal email {e}" for e in emails if not e.endswith(NOREPLY)]
            hits += [f"{c[:7]}: {what}" for what in private_hits(ident.replace("<", " ").replace(">", " "))
                     if "email" not in what]
            for l in msg.splitlines():
                if "noreply@anthropic.com" in l:  # co-author trailer
                    continue
                hits += [f"{c[:7]} message: {what}: {l.strip()[:100]}" for what in private_hits(l)]
        self.assertEqual(hits, [], "\n".join(hits))


class ReadmeFacts(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "models.json"), encoding="utf-8") as f:
            self.models = json.load(f)["models"]
        self.text = readme()

    def test_every_manifest_file_is_in_the_readme_with_its_exact_size(self):
        for m in self.models:
            row = next((l for l in self.text.splitlines() if f"`{m['path']}`" in l), None)
            self.assertIsNotNone(row, f"{m['path']} missing from README model table")
            self.assertIn(f"{m['bytes']:,}", row, m["path"])
            self.assertIn(m["url"], row, m["path"])

    def test_readme_totals_match_the_manifest(self):
        def total(roles):
            return f"{sum(m['bytes'] for m in self.models if m['role'] in roles):,}"
        self.assertIn(f"**{total({'depth'})}**", self.text)
        self.assertIn(f"**{total({'depth', 'cinema'})}**", self.text)
        self.assertIn(f"**{total({m['role'] for m in self.models})}**", self.text)

    def test_every_repo_file_the_readme_names_exists(self):
        model_files = {m["path"].rsplit("/", 1)[1] for m in self.models}
        runtime_outputs = {"burst-ready.txt", "burst-done.txt", "burst-log.csv", "capture-status.txt", "process.json"}
        # extra_model_paths.yaml: the name ComfyUI expects; we ship the .example
        not_ours = model_files | runtime_outputs | {"extra_model_paths.yaml"}
        names = set(re.findall(r"`((?:[\w.-]+/)*[\w.-]+\.(?:py|json|yaml|mp4|txt|ps1))`", self.text))
        for name in sorted(names):
            if name.startswith("models/") or name in not_ours:
                continue
            self.assertTrue(os.path.exists(os.path.join(ROOT, name)), f"README names missing file {name}")

    def test_documented_run_local_commands_parse(self):
        import shlex
        import sys
        sys.path.insert(0, ROOT)
        import run_local
        cmds = re.findall(r"^\s*python run_local\.py ([^\n#]+)", self.text, re.M)
        self.assertGreaterEqual(len(cmds), 2, "README should show depth and cinema commands")
        parser = run_local.build_parser()
        for cmd in cmds:
            parser.parse_args(shlex.split(cmd))  # SystemExit (test error) on a bad flag


if __name__ == "__main__":
    unittest.main()
