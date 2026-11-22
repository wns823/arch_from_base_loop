"""Upload one harness agent-call record (runs/<case>/<run>/calls/<seq>_<role>.json) to W&B Weave.

Run by driver.finish_call in the background with a Python that has `weave` (default ~/miniforge3/bin/python), so the
harness venv needs no extra packages. One Weave call per agent call (inputs: prompt, decision, memory file; output:
reply, outcome, cost), with one child call per tool use (input: tool arguments; output: tool result).
Project: $AGH_WEAVE_PROJECT (default <entity of the Claude Code Weave plugin>/ag-harness).
API key: $WANDB_API_KEY, else the key the Weave Claude Code plugin already stores in ~/.weave-claude-code/settings.json.
"""
import json
import os
import sys
from pathlib import Path

CLIP = 20000   # characters per text field, to keep traces light


def clip(x):
    if isinstance(x, str):
        return x if len(x) <= CLIP else x[:CLIP] + f"... [{len(x) - CLIP} chars clipped]"
    if isinstance(x, list):
        return [clip(v) for v in x]
    if isinstance(x, dict):
        return {k: clip(v) for k, v in x.items()}
    return x


def main(path):
    path = Path(path)
    rec = json.load(open(path))
    if rec.get("weave_call_id"):
        return
    plug = {}
    pf = Path.home() / ".weave-claude-code" / "settings.json"
    if pf.exists():
        plug = json.load(open(pf))
    if not os.environ.get("WANDB_API_KEY"):
        key = plug.get("wandb_api_key") or plug.get("api_key")
        if key:
            os.environ["WANDB_API_KEY"] = key
    entity = (plug.get("weave_project") or "").split("/")[0]
    project = os.environ.get("AGH_WEAVE_PROJECT") or (f"{entity}/ag-harness" if entity else "ag-harness")
    import weave
    client = weave.init(project)

    meta = rec.get("meta") or {}
    cand = meta.get("cand") or (f"space_v{meta['space_version']}" if "space_version" in meta else "")
    name = f"{rec.get('case')}/{rec.get('run')} {rec['role']} {cand}".strip()
    attrs = {"case": rec.get("case"), "run": rec.get("run"), "role": rec["role"], "seq": rec["seq"],
             "cand": meta.get("cand"), "attempt": meta.get("attempt"), "session_id": rec.get("session_id")}
    parent = client.create_call(f"agh_{rec['role'].replace('-', '_')}",
                                inputs=clip({"prompt": rec["prompt"], "meta": meta, "memory": rec.get("memory")}),
                                attributes=attrs, display_name=name)
    pending = {}
    for e in rec.get("transcript", []):
        content = (e.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if b.get("type") == "tool_use":
                pending[b["id"]] = client.create_call(f"tool_{b.get('name')}", inputs=clip(b.get("input") or {}),
                                                      parent=parent, display_name=b.get("name"))
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in pending:
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                client.finish_call(pending.pop(b["tool_use_id"]), output=clip({"result": c, "is_error": b.get("is_error")}))
            elif b.get("type") == "text" and e.get("type") == "assistant":
                client.finish_call(client.create_call("assistant_text", inputs={}, parent=parent),
                                   output=clip({"text": b.get("text")}))
    for c in pending.values():
        client.finish_call(c, output={"result": None})
    client.finish_call(parent, output=clip({"reply": rec.get("reply"), "outcome": rec.get("outcome"),
                                            "cost_usd": rec.get("cost_usd"), "usage": rec.get("usage"),
                                            "secs": rec.get("secs"), "is_error": rec.get("is_error")}))
    client.flush()
    rec["weave_call_id"] = parent.id
    rec["weave_project"] = project
    tmp = Path(str(path) + ".tmp")
    json.dump(rec, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
