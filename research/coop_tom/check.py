"""Check a served model before an experiment: is thinking really off, and does JSON come back?

Qwen3.x chat templates switch thinking with `enable_thinking`; the Qwen3.8 guide mentions
`reasoning_effort: none`. Which one the served template honours is checked here instead of
assumed. Each candidate `chat_template_kwargs` is sent with a tiny JSON task; a candidate
passes when no reasoning text comes back (neither `reasoning_content` nor inline <think>)
and the reply is the requested JSON.

  python -m research.coop_tom.check --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b
  python -m research.coop_tom.check ... --pick   # print only the first passing JSON (for scripts)

Exit code 1 if no candidate passes.
"""
import argparse
import json
import sys

from .llm import OpenAICompatBackend, parse_json

CANDIDATES = [
    {"enable_thinking": False, "reasoning_effort": "none"},
    {"enable_thinking": False},
    {"reasoning_effort": "none"},
]
TASK = [{"role": "user", "content": 'Reply with JSON only: {"action": "wait 1"}'}]


def probe(backend: OpenAICompatBackend, kwargs: dict, max_tokens: int = 256) -> dict:
    backend.extra_body = {"chat_template_kwargs": kwargs}
    r = backend.chat(TASK, max_tokens=max_tokens, seed=0)
    js = parse_json(r.text)
    return {"chat_template_kwargs": kwargs, "error": r.error, "thinking": r.thinking,
            "reasoning_chars": len(r.reasoning), "completion_tokens": r.completion_tokens,
            "finish_reason": r.finish_reason, "json_ok": bool(js) and js.get("action") == "wait 1",
            "content": r.text[:120],
            "passes": not r.error and not r.thinking and bool(js) and js.get("action") == "wait 1"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="ICRL_LLM_API_KEY")
    ap.add_argument("--pick", action="store_true", help="print only the first passing kwargs as JSON")
    a = ap.parse_args(argv)
    backend = OpenAICompatBackend("check", a.base_url, a.model, api_key_env=a.api_key_env,
                                  params={"temperature": 0.0}, max_retries=1)
    served = backend.models()
    if a.model not in served:
        sys.exit(f"model {a.model!r} not served at {a.base_url}; available {served}")
    results = []
    for kwargs in CANDIDATES:
        results.append(probe(backend, kwargs))
        if a.pick and results[-1]["passes"]:
            break
    # Sanity check of the detector: asking for thinking should produce reasoning.
    on = probe(backend, {"enable_thinking": True, "reasoning_effort": "high"}, max_tokens=512)
    if not a.pick:
        for r in results + [dict(on, note="thinking requested ON (detector sanity check)")]:
            print(json.dumps(r))
    passing = [r for r in results if r["passes"]]
    if not passing:
        print("No candidate turns thinking off with valid JSON; see the lines above.", file=sys.stderr)
        sys.exit(1)
    if not on["thinking"]:
        print("warning: thinking requested ON produced no reasoning text; the server may lack a "
              "reasoning parser, or the template ignores these switches.", file=sys.stderr)
    if a.pick:
        print(json.dumps(passing[0]["chat_template_kwargs"]))


if __name__ == "__main__":
    main()
