"""Verify what actually gets passed to Ollama and whether it fits in context."""
import json
import os

work_dir = "recordings/Mar27th/.narrator_work"

with open(os.path.join(work_dir, "cleaned_segments.json")) as f:
    segs = json.load(f)

plain_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in segs)
chars = len(plain_text)
approx_tokens = chars // 4

print(f"Segments: {len(segs)}")
print(f"Plain-text chars: {chars:,}")
print(f"Approx tokens: {approx_tokens:,}")

# How much fits in common context windows (reserve 500 tokens for system prompt + response)
for limit_name, limit in [
    ("llama3 default (4K ctx)", 4096),
    ("llama3.1/3.2 (128K ctx)", 131072),
    ("gemini-1.5-pro (1M ctx)", 1_000_000),
]:
    usable_chars = (limit - 500) * 4
    running = 0
    fit_segs = 0
    for s in segs:
        line = f"{s['speaker']}: {s['text']}\n"
        if running + len(line) > usable_chars:
            break
        running += len(line)
        fit_segs += 1
    last_fit = segs[fit_segs - 1] if fit_segs else None
    end_time = f"{int(last_fit['end'])//60}m{int(last_fit['end'])%60}s" if last_fit else "0"
    print(f"  {limit_name}: fits {fit_segs}/{len(segs)} segments (up to {end_time})")

last_end = max(s["end"] for s in segs)
print(f"\nTotal session duration: {int(last_end)//3600}h {(int(last_end)%3600)//60}m")

print("\n=== FIRST 5 SEGMENTS ===")
for s in segs[:5]:
    print(f"  [{int(s['start'])//60:02d}:{int(s['start'])%60:02d}] {s['speaker']}: {s['text']}")

print("\n=== LAST 5 SEGMENTS ===")
for s in segs[-5:]:
    print(f"  [{int(s['start'])//60:02d}:{int(s['start'])%60:02d}] {s['speaker']}: {s['text']}")


with open(os.path.join(work_dir, "transcript.json")) as f:
    transcript = json.load(f)
with open(os.path.join(work_dir, "cleaned_segments.json")) as f:
    cleaned = json.load(f)

t_segs = transcript if isinstance(transcript, list) else transcript.get("segments", transcript)
c_segs = cleaned if isinstance(cleaned, list) else cleaned.get("segments", cleaned)

print(f"Transcript segments: {len(t_segs)}")
print(f"Cleaned segments: {len(c_segs)}")

t_text = " ".join(s["text"] for s in t_segs)
c_text = " ".join(s["text"] for s in c_segs)
print(f"Transcript total chars: {len(t_text)}")
print(f"Cleaned total chars: {len(c_text)}")
print(f"Are they identical? {t_text == c_text}")

diffs = [
    (i, t_segs[i]["text"], c_segs[i]["text"])
    for i in range(min(len(t_segs), len(c_segs)))
    if t_segs[i]["text"] != c_segs[i]["text"]
]
print(f"Segments that differ: {len(diffs)}")
for i, t, c in diffs[:10]:
    print(f"  [{i}] ORIG:  {t!r}")
    print(f"  [{i}] CLEAN: {c!r}")

# Show a slice from the middle of the session to check content
print("\n=== TRANSCRIPT SLICE (segments 400-420) ===")
for s in t_segs[400:420]:
    print(f"  {s['speaker']}: {s['text']}")

print("\n=== CLEANED SLICE (segments 400-420) ===")
for s in c_segs[400:420]:
    print(f"  {s['speaker']}: {s['text']}")

# Count how much text is actually sent to writer
full_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in c_segs)
print(f"\nFull cleaned text length: {len(full_text)} chars (~{len(full_text)//4} tokens)")
