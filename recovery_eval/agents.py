"""Bounded, observable language-model agents that execute real simulator tools.

Backends share a plain-text JSON action protocol. No native function-calling
support is assumed, no answer is silently repaired, and no model is loaded or
remote request sent at import time. Rule-policy results are never mixed into
language-model episodes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


class ModelAdapterError(RuntimeError):
    """A model infrastructure error, separate from an agent task failure."""

    def __init__(self, message: str, *, code: str = "backend_error"):
        super().__init__(message)
        self.code = code


@dataclass
class ModelResponse:
    """One unmodified model completion and its available runtime evidence."""

    text: str
    generated_token_ids: list[int] | None = None
    latency_seconds: float = 0.0
    usage: dict[str, int | float | None] = field(default_factory=dict)
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    """Small backend interface; one instance is reused across all episodes."""

    model_id: str
    kind: str

    def complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        ...

    def describe(self) -> dict[str, Any]:
        ...


def _positive_integer(value: Any, name: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer between 1 and {maximum or 'unbounded'}")
    return value


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class LocalTransformersAdapter:
    """Lazy-loaded CPU-only greedy generation, with immutable revision evidence."""

    kind = "local_model"

    def __init__(self, model_id: str = DEFAULT_MODEL, *, revision: str = "main",
                 cache_dir: str | Path | None = None, max_new_tokens: int = 96,
                 threads: int = 4, local_files_only: bool = True):
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.max_new_tokens = _positive_integer(max_new_tokens, "max_new_tokens", 4096)
        self.threads = _positive_integer(threads, "threads", 256)
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._revision_sha: str | None = None
        self._load_seconds: float | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # Heavyweight libraries stay optional for offline rule-policy use.
        import torch
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(self.threads)
        torch.manual_seed(0)
        options = {"revision": self.revision, "cache_dir": self.cache_dir,
                   "local_files_only": self.local_files_only, "token": False,
                   "trust_remote_code": False}
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **options)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.float32, **options
        ).to("cpu").eval()
        revision_sha = getattr(model.config, "_commit_hash", None)
        if not revision_sha:
            cached = try_to_load_from_cache(
                self.model_id, "config.json", revision=self.revision,
                cache_dir=self.cache_dir,
            )
            if isinstance(cached, str):
                revision_sha = Path(cached).parent.name
        if not (isinstance(revision_sha, str) and len(revision_sha) == 40
                and all(char in "0123456789abcdef" for char in revision_sha.lower())):
            raise ModelAdapterError("Cannot resolve an immutable model revision SHA.", code="unresolved_revision")
        self._revision_sha = revision_sha
        self._model, self._tokenizer, self._torch = model, tokenizer, torch
        self._load_seconds = time.perf_counter() - started

    def complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        self._load()
        tokenizer, model, torch = self._tokenizer, self._model, self._torch
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_count = int(encoded["input_ids"].shape[-1])
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit and input_count + self.max_new_tokens > context_limit:
            raise ModelAdapterError("Input exceeds model context; no truncation was applied.", code="context_overflow")
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **encoded, do_sample=False, max_new_tokens=self.max_new_tokens,
                pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                              else tokenizer.eos_token_id), use_cache=True,
            )
        latency = time.perf_counter() - started
        generated_ids = outputs[0, input_count:].tolist()
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        ended = bool(generated_ids and generated_ids[-1] in eos)
        at_limit = len(generated_ids) >= self.max_new_tokens
        return ModelResponse(
            text=tokenizer.decode(generated_ids, skip_special_tokens=True),
            generated_token_ids=generated_ids, latency_seconds=latency,
            usage={"prompt_tokens": input_count, "completion_tokens": len(generated_ids),
                   "total_tokens": input_count + len(generated_ids)},
            finish_reason="eos" if ended else ("max_new_tokens" if at_limit else "other"),
            metadata={"generation_truncated": at_limit and not ended,
                      "finish_reason_source": "generated_token_ids", "input_truncated": False},
        )

    def describe(self) -> dict[str, Any]:
        return {"backend": "transformers_cpu", "model_id": self.model_id,
                "requested_revision": self.revision,
                "resolved_revision_sha": self._revision_sha,
                "device": "cpu", "dtype": "float32", "do_sample": False,
                "max_new_tokens": self.max_new_tokens, "threads": self.threads,
                "batch_size": 1, "seed": 0, "local_files_only": self.local_files_only,
                "model_load_seconds": self._load_seconds,
                "versions": {name: _package_version(name) for name in
                             ("torch", "transformers", "huggingface_hub", "tokenizers")}}


class OpenAICompatibleAdapter:
    """Opt-in text completions over an OpenAI-compatible chat endpoint.

    Credentials are read only when complete() is called. Endpoint errors do not
    retain response bodies, authorization headers, or exception text. There are
    no implicit request retries. Provider token IDs are unavailable by default.
    """

    kind = "api_model"

    def __init__(self, model_id: str, base_url: str, *, api_key_env: str = "PRIME_API_KEY",
                 max_new_tokens: int = 128, timeout_seconds: float = 60):
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in ("https", "http") or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Unencrypted HTTP endpoints are allowed only on localhost")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ValueError("api_key_env must name a credential environment variable")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.max_new_tokens = _positive_integer(max_new_tokens, "max_new_tokens", 4096)
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ModelAdapterError("Required API credential environment variable is absent; no request sent.", code="missing_credential")
        payload = {"model": self.model_id, "messages": messages,
                   "max_tokens": self.max_new_tokens, "temperature": 0}
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.load(response)
            choice = data["choices"][0]
            raw = choice["message"]["content"]
            if not isinstance(raw, str):
                raise ValueError("Expected a text response")
            provider_usage = data.get("usage") or {}
            if not isinstance(provider_usage, dict):
                raise ValueError("Invalid usage object")
            usage = {name: provider_usage.get(name) for name in
                     ("prompt_tokens", "completion_tokens", "total_tokens")}
            if any(value is not None and (type(value) not in (int, float) or value < 0)
                   for value in usage.values()):
                raise ValueError("Invalid token usage")
        except urllib.error.HTTPError as error:
            raise ModelAdapterError(f"Model endpoint returned HTTP {error.code}.", code="http_error") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ModelAdapterError("Model endpoint request failed.", code="network_error") from None
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ModelAdapterError("Model endpoint returned an invalid text completion.", code="invalid_provider_response") from None
        return ModelResponse(
            text=raw, latency_seconds=time.perf_counter() - started,
            generated_token_ids=None, usage=usage,
            finish_reason=choice.get("finish_reason"),
            metadata={"token_ids_available": False,
                      "token_ids_note": "The chat endpoint does not expose generated token IDs.",
                      "finish_reason_source": "provider_response"},
        )

    def describe(self) -> dict[str, Any]:
        return {"backend": "openai_compatible_text", "model_id": self.model_id,
                "base_url": self.base_url, "max_new_tokens": self.max_new_tokens,
                "temperature": 0, "timeout_seconds": self.timeout_seconds,
                "native_function_calling": False, "implicit_network_retries": 0,
                "credentials_logged": False}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON constant: {value}")


def parse_action(raw: str) -> dict[str, Any]:
    """Parse the entire response strictly: no fences, extraction, or coercion."""
    if not isinstance(raw, str):
        raise ValueError("The model completion must be text")
    action = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(action, dict):
        raise ValueError("An action must be a JSON object")
    if set(action) == {"tool", "arguments"}:
        if not isinstance(action["tool"], str) or not action["tool"] or not isinstance(action["arguments"], dict):
            raise ValueError("A tool action requires a nonempty tool name and object arguments")
        return action
    if set(action) == {"finish"} and isinstance(action["finish"], dict):
        final = action["finish"]
        if (set(final) in ({"success"}, {"success", "message"})
                and type(final.get("success")) is bool
                and ("message" not in final or isinstance(final["message"], str))):
            return action
    raise ValueError("Expected exactly {tool, arguments} or {finish: {success, optional message}}")


def build_messages(public_task: dict[str, Any], tool_schemas: list[dict[str, Any]],
                   prompt_variant: str = "basic") -> list[dict[str, str]]:
    """Use public task/tool contracts only, never case metadata or oracle state."""
    if prompt_variant not in ("basic", "guarded"):
        raise ValueError("prompt_variant must be basic or guarded")
    instruction = (
        "Execute the requested workflow using tools. Return only ONE JSON object per turn: "
        '{"tool":"tool_name","arguments":{...}} or {"finish":{"success":true}}. '
        "Use false if you cannot establish completion. No prose, Markdown, or multiple actions. "
        "Tool responses arrive as user messages with an observation field. "
        "Follow the public step plan, use exact task values, and complete every requested effect exactly once."
    )
    if prompt_variant == "guarded":
        instruction += (
            " A write timeout means its outcome is UNKNOWN. Reuse a stable operation_id for each "
            "intended write; never change its payload on retry. Recheck committed evidence when needed. "
            "A not_found result may be stale and does not cancel pending writes. Only claim success "
            "when tool observations establish all required effects."
        )
    # The schema is public for all compared agents; the guarded variant adds advice only.
    instruction += "\nAvailable tool contracts: " + json.dumps(tool_schemas, separators=(",", ":"), ensure_ascii=False)
    return [{"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(public_task, separators=(",", ":"), ensure_ascii=False)}]


def run_agent(case: dict[str, Any], adapter: ModelAdapter, *, prompt_variant: str = "basic",
              max_turns: int = 8, max_tool_calls: int = 12, format_retries: int = 1) -> dict[str, Any]:
    """Run a real model-in-the-loop recovery episode with independent scoring.

    Every model request counts against max_turns, including a format correction.
    The optional correction is explicit in messages and model_calls. At most one
    correction is allowed per episode. The oracle is consulted only by finish().
    """
    from .engine import Environment

    _positive_integer(max_turns, "max_turns", 100)
    _positive_integer(max_tool_calls, "max_tool_calls", 100)
    if type(format_retries) is not int or format_retries not in (0, 1):
        raise ValueError("format_retries must be 0 or 1")
    environment = Environment(case, max_calls=max_tool_calls)
    messages = build_messages(environment.public_task, environment.tool_schemas, prompt_variant)
    prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    model_calls: list[dict[str, Any]] = []
    claimed_success = False
    status = "turn_budget_exhausted"
    tool_calls = 0
    corrections = 0
    correction_request = False
    for turn in range(1, max_turns + 1):
        started = time.perf_counter()
        try:
            response = adapter.complete(deepcopy(messages))
            if not isinstance(response, ModelResponse):
                raise TypeError("Adapters must return ModelResponse")
        except Exception as error:
            # Exception text may contain credentials; retain type and allowlisted code only.
            safe_codes = {"backend_error", "unresolved_revision", "context_overflow", "missing_credential",
                          "http_error", "network_error", "invalid_provider_response"}
            error_code = error.code if isinstance(error, ModelAdapterError) and error.code in safe_codes else "unclassified_backend_error"
            model_calls.append({"turn": turn, "status": "infrastructure_error",
                                "error_type": type(error).__name__,
                                "error_code": error_code,
                                "latency_seconds": time.perf_counter() - started,
                                "format_correction_request": correction_request})
            status = "infrastructure_error"
            break
        call = {"turn": turn, "status": "received", "raw_generation": response.text,
                "generated_token_ids": response.generated_token_ids,
                "latency_seconds": response.latency_seconds, "usage": deepcopy(response.usage),
                "finish_reason": response.finish_reason, "metadata": deepcopy(response.metadata),
                "format_correction_request": correction_request}
        correction_request = False
        model_calls.append(call)
        messages.append({"role": "assistant", "content": response.text})
        try:
            action = parse_action(response.text)
        except (ValueError, TypeError) as error:
            call.update({"status": "invalid_action", "parse_error": str(error)})
            if corrections < format_retries:
                corrections += 1
                messages.append({"role": "user", "content": (
                    "FORMAT ERROR. Your entire previous reply was not a valid action. "
                    'Reply only {"tool":"tool_name","arguments":{...}} or '
                    '{"finish":{"success":true}} using a JSON boolean. Do not explain.'
                )})
                correction_request = True
                continue
            status = "invalid_action"
            break
        call["parsed_action"] = action
        if "finish" in action:
            claimed_success = action["finish"]["success"]
            call["status"] = "finish"
            status = "completed"
            break
        if tool_calls >= max_tool_calls:
            call["status"] = "tool_budget_exhausted"
            status = "tool_budget_exhausted"
            break
        result = environment.invoke(action["tool"], action["arguments"])
        tool_calls += 1
        call.update({"status": "tool_executed", "tool_response": deepcopy(result)})
        messages.append({"role": "user", "content": json.dumps(
            {"observation": {"tool": action["tool"], "response": result}},
            separators=(",", ":"), ensure_ascii=False,
        )})
    row = environment.finish(
        claimed_success, agent_id=f"{adapter.model_id}:{prompt_variant}", agent_kind=adapter.kind
    )
    row.update({"evaluation_mode": "live_model_recovery", "model": adapter.model_id,
                "model_backend": adapter.describe(), "prompt_variant": prompt_variant,
                "prompt_hash": prompt_hash, "run_status": status, "model_calls": model_calls,
                "messages": messages, "max_turns": max_turns, "max_tool_calls": max_tool_calls,
                "format_retry_limit": format_retries, "format_corrections": corrections,
                "model_call_count": len(model_calls), "actual_tool_invocations": tool_calls,
                "labels_in_prompt": False, "silent_output_repair": False,
                "input_truncation_applied": False})
    return row
