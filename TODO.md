# Security Audit — TODO

Audit performed 2026-03-30 against the untrusted codebase.

---

## HIGH

### [x] Prompt Injection via Transcript Content (`writer.py`)
Transcript text was interpolated directly into LLM prompts via `.format()`.
Audio content saying "ignore previous instructions…" would be injected verbatim.
**Fix:** Restructured backend helpers to accept a `system` + `user` argument
pair and pass the transcript only as the user-role message, keeping
instructions in the system role (Ollama `messages=[…]`, Gemini
`system_instruction=`).

---

## MEDIUM

### [x] ffmpeg Concat List Injection via Malicious Filenames (`joiner.py`)
Single-quote-wrapped paths in the ffmpeg concat list were not escaped.
A filename containing `'` or a literal newline could inject extra `file`
directives.
**Fix:** Escape `\` then `'` in every resolved path before writing the
concat list.

### [x] API Key Exposed in Process List (`cli.py`)
`--gemini-api-key` passed as a CLI flag appeared in `ps aux` and shell
history.
**Fix:** Removed the `--gemini-api-key` CLI option from both `narrate` and
`write` subcommands. The key is now read exclusively from the
`GEMINI_API_KEY` environment variable.

---

## LOW

### [x] Unchecked `json.loads()` on LLM Response (`writer.py`)
No size guard before parsing the LLM response, allowing excessive memory
use from a manipulated/verbose reply.
**Fix:** Added a length guard — if the response is more than 10× the size
of the input transcript, skip parsing and fall back to the original
segments.

### [x] `shutil.rmtree` with `ignore_errors=True` on Possibly-Symlinked
`work_dir` (`cli.py`)
A symlink planted at the default work dir location could cause a silent
recursive delete of an unintended directory.
**Fix:** Check for a symlink before deletion and refuse to `rmtree` it;
also removed `ignore_errors=True` so real failures surface.

---

## INFORMATIONAL

### [x] No Pin on Dependency Versions
`requirements.txt` and `pyproject.toml` use `>=` lower-bounds only.
A compromised upstream release of `whisperx`, `google-generativeai`, or
`ollama` would be silently installed.
**Recommended fix (manual):**
```
pip install pip-tools
pip-compile --generate-hashes requirements.txt -o requirements.lock.txt
# Then install with:
pip install -r requirements.lock.txt
```
