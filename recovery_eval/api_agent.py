"""Optional bounded OpenAI-compatible live-agent runner.

No API calls occur at import time. Keys are read from the named environment
variable and are never written into the result bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

from .core import RecoveryEnvironment, TOOL_SCHEMAS, build_cases
from .reporting import write_artifacts


def request_chat(base_url: str, key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"model endpoint returned HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("model endpoint network request failed") from None


def run_live_case(case: dict, model: str, base_url: str, key: str,
                  prompt_variant: str = "basic", max_turns: int = 12) -> dict:
    if not 1 <= max_turns <= 100:
        raise ValueError("max_turns must be between 1 and 100")
    environment = RecoveryEnvironment(case)
    instruction = (
        "Complete the requested workflow using the provided tools. Create exactly one correct record "
        "and exactly one notification linked to it. Do not omit work or duplicate effects. "
        "When finished, return only JSON: {\"success\": true} or {\"success\": false}."
    )
    if prompt_variant == "guarded":
        instruction += (
            " A write timeout has an unknown outcome. Use a stable operation_id for each intended "
            "write and reuse it unchanged on retries. A not_found read may be stale; it does not "
            "cancel pending writes. Stop honestly if completion cannot be established."
        )
    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(environment.task)}]
    calls = []
    claimed = False
    status = "turn_budget_exhausted"
    total_tool_calls = 0
    for _ in range(max_turns):
        started = time.perf_counter()
        try:
            response = request_chat(base_url, key, {"model": model, "messages": messages,
                                    "tools": TOOL_SCHEMAS, "max_tokens": 512})
            message = response["choices"][0]["message"]
            if not isinstance(message, dict) or not isinstance(message.get("tool_calls") or [], list):
                raise ValueError("malformed model message")
        except (RuntimeError, KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
            status = "infrastructure_error"
            calls.append({"error_type": type(exc).__name__, "latency_seconds": time.perf_counter() - started})
            break
        calls.append({"message": message, "usage": response.get("usage", {}),
                      "latency_seconds": time.perf_counter() - started})
        tool_calls = message.get("tool_calls") or []
        if total_tool_calls + len(tool_calls) > 24:
            status = "tool_budget_exhausted"
            break
        assistant = {"role": "assistant", "content": message.get("content")}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            try:
                answer = json.loads(message.get("content") or "")
                if set(answer) != {"success"} or type(answer["success"]) is not bool:
                    raise ValueError("invalid final answer")
                claimed = answer["success"]
                status = "completed"
            except (TypeError, ValueError, KeyError):
                status = "invalid_final_answer"
            break
        for tool in tool_calls:
            try:
                if (not isinstance(tool, dict) or not isinstance(tool.get("id"), str)
                        or not isinstance(tool.get("function"), dict)):
                    status = "invalid_tool_call"
                    break
                args = json.loads(tool["function"]["arguments"])
                result = environment.invoke(tool["function"]["name"], args)
            except (TypeError, ValueError, KeyError):
                result = {"ok": False, "error": {"type": "invalid_tool_call"}}
            total_tool_calls += 1
            messages.append({"role": "tool", "tool_call_id": tool["id"], "content": json.dumps(result)})
        if status == "invalid_tool_call":
            break
    row = environment.finish(claimed, f"{model}:{prompt_variant}", evaluation_mode="live_model")
    row.update({"model": model, "model_calls": calls, "messages": messages, "run_status": status,
                "prompt_variant": prompt_variant, "max_turns": max_turns})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional real model recovery evaluation; requires your model endpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="PRIME_API_KEY")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--prompt-variant", choices=("basic", "guarded"), default="basic")
    parser.add_argument("--out", default="results/api_agent")
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env)
    if not key:
        parser.error(f"required environment variable {args.api_key_env} is absent; no requests were sent")
    if args.limit < 1 or args.limit > 60:
        parser.error("limit must be between 1 and 60")
    cases = build_cases(12, 1)[:args.limit]
    results = [run_live_case(c, args.model, args.base_url, key, args.prompt_variant) for c in cases]
    write_artifacts(results, args.out, metadata={"backend": "external_model_endpoint",
                    "experiment_kind": "live_model_recovery", "model": args.model})
    print(json.dumps({"episodes": len(results), "report": args.out + "/report.html"}))


if __name__ == "__main__":
    main()
