"""
Patch async_omni_engine.py to include the real exception + traceback in
the "Orchestrator thread crashed" error message, so it surfaces via the
HTTP 500 response instead of being silently swallowed into container logs.
"""
filepath = "/usr/local/lib/python3.12/dist-packages/vllm_omni/engine/async_omni_engine.py"

with open(filepath, "r") as f:
    content = f.read()

old = 'self.output_queue.sync_q.put_nowait({"type": "error", "error": "Orchestrator thread crashed"})'
new = 'self.output_queue.sync_q.put_nowait({"type": "error", "error": repr(e) + chr(10) + __import__(\'traceback\').format_exc()})'

if old in content:
    content = content.replace(old, new)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"[patch_omni] Patched {filepath}")
else:
    # Show context so we can adjust the pattern if needed
    idx = content.find("Orchestrator thread crashed")
    print(f"[patch_omni] WARNING: exact pattern not found. "
          f"'Orchestrator thread crashed' at idx={idx}")
    if idx >= 0:
        print("Context:", repr(content[max(0, idx - 120): idx + 200]))
