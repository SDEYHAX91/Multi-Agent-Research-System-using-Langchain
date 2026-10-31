import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

load_dotenv()

# ------------------------------------------------------------------ #
# Logging (plain ASCII so it is safe on Windows consoles)
# ------------------------------------------------------------------ #
logger = logging.getLogger("llm_failsafe")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ------------------------------------------------------------------ #
# Providers and default fail-over chain
# ------------------------------------------------------------------ #
# env: list of "any-of" groups - every group needs at least one variable set.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {"env": [["GROQ_API_KEY"]], "pkg": "langchain-groq"},
    "mistralai": {"env": [["MISTRAL_API_KEY"]], "pkg": "langchain-mistralai"},
    "openrouter": {"env": [["OPENROUTER_API_KEY"]], "pkg": "langchain-openrouter"},
    "ollama": {"env": [], "pkg": "langchain-ollama"},
}

@dataclass(eq=False)
class ModelSpec:
    provider: str
    model: str
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)  # extra kwargs for init_chat_model
    builder: Optional[Callable[["ModelSpec", float], BaseChatModel]] = None  # custom constructor

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


def _build_mistral(spec: ModelSpec, temperature: float) -> BaseChatModel:
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(model=spec.model, temperature=temperature)


# Order = priority. The first usable model is always tried first, so the app
# returns to the primary model as soon as its cooldown has expired.
DEFAULT_CHAIN: List[ModelSpec] = [
    ModelSpec("mistralai", "ministral-14b-2512", builder=_build_mistral),
    ModelSpec("groq", "openai/gpt-oss-120b"),
    ModelSpec("openrouter", "openai/gpt-oss-120b"),
    ModelSpec("ollama", "nemotron-3-super:cloud"),
    ModelSpec("ollama", "gemma4:31b-cloud"),
]


# ------------------------------------------------------------------ #
# Per-role assignment: preferred model first, then the best alternatives
# ------------------------------------------------------------------ #
_MODELS_BY_NAME = {
    "gpt20_groq": ("groq", "openai/gpt-oss-20b"),
    "gpt20_or": ("openrouter", "openai/gpt-oss-20b"),
    "ministral": ("mistralai", "ministral-14b-2512"),
    "gpt120_groq": ("groq", "openai/gpt-oss-120b"),
    "gpt120_or": ("openrouter", "openai/gpt-oss-120b"),
    "nemotron_ollama": ("ollama", "nemotron-3-super:cloud"),
    "gemma4_ollama": ("ollama", "gemma4:31b-cloud"),
}


def _chain(*names: str) -> List[ModelSpec]:
    specs = []
    for n in names:
        provider, model = _MODELS_BY_NAME[n]
        if provider == "mistralai":
            specs.append(ModelSpec(provider, model, builder=_build_mistral))
        else:
            specs.append(ModelSpec(provider, model))
    return specs


ROLE_CHAINS: Dict[str, List[ModelSpec]] = {
    # Preferred model first; later entries are used only if it fails.
    "search": _chain("ministral", "gpt20_groq"),
    "reader": _chain("gpt120_groq", "gpt20_groq"),
    "writer": _chain("gpt120_or", "gpt20_or"),
    "critic": _chain("nemotron_ollama", "gemma4_ollama"),
}


def chain_from_env(value: str) -> List[ModelSpec]:
    """Parse 'provider:model,provider:model' (first ':' splits provider and model)."""
    specs = []
    for item in value.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        provider, model = item.split(":", 1)
        provider = provider.strip().lower().replace("-", "_")
        specs.append(ModelSpec(provider, model.strip()))
    return specs


# ------------------------------------------------------------------ #
# Error classification
# ------------------------------------------------------------------ #
# seconds a model is parked after a failure of this kind
COOLDOWNS = {
    "rate_limit": 60,      # token / request / quota limits (overridden by "retry in Xs" hints)
    "auth": 3600,          # bad / missing key
    "not_found": 1800,     # model removed / not available / tools unsupported
    "transient": 20,       # timeouts, 5xx, overloaded
    "other": 30,
    "too_large": 0,        # request bigger than this model accepts -> only skip for this call
    "context_length": 0,   # prompt longer than this model's context -> only skip for this call
}


def _status_code(exc: BaseException) -> Optional[int]:
    for obj in (exc, getattr(exc, "response", None)):
        if obj is None:
            continue
        for attr in ("status_code", "http_status", "status", "code"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return value
    return None


def classify_error(exc: BaseException) -> str:
    code = _status_code(exc)
    text = f"{type(exc).__name__}: {exc}".lower()

    if any(k in text for k in (
        "context length", "context_length", "maximum context", "context window",
        "prompt is too long", "input is too long", "reduce the length",
    )):
        return "context_length"
    if code == 413 or "request too large" in text or "payload too large" in text:
        return "too_large"
    if code == 429 or any(k in text for k in (
        "rate limit", "rate_limit", "ratelimit", "resource_exhausted", "resourceexhausted",
        "quota", "too many requests", "tokens per", "requests per", "insufficient_quota",
    )):
        return "rate_limit"
    if code in (401, 403) or any(k in text for k in (
        "api key", "api_key", "unauthorized", "unauthenticated", "permission denied", "authentication",
    )):
        return "auth"
    if code == 404 or any(k in text for k in (
        "not found", "does not exist", "decommissioned", "deprecated", "unsupported model", "not supported",
    )):
        return "not_found"
    if code in (408, 500, 502, 503, 504, 529) or any(k in text for k in (
        "timeout", "timed out", "connection", "overloaded", "unavailable", "temporarily",
        "internal server error", "bad gateway",
    )):
        return "transient"
    return "other"


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Pull a 'try again in 1m4.3s' style hint out of headers / the error text."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?:try again in|retry in|retrydelay\W+)\s*([0-9hms.\s]+)", str(exc).lower())
    if not match:
        return None
    total, found = 0.0, False
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", match.group(1)):
        found = True
        total += float(number) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def _short(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}"[:limit]


class AllModelsFailedError(RuntimeError):
    """Raised when every usable model in the chain failed."""

    def __init__(self, errors: List[Tuple[str, str, str]]):
        self.errors = errors
        lines = "\n".join(f"  - {key}: [{cat}] {msg}" for key, cat, msg in errors)
        if not lines:
            lines = "  (no usable models - check API keys / installed packages: python llm_failsafe.py)"
        super().__init__("All configured models failed:\n" + lines)


# ------------------------------------------------------------------ #
# Content normalisation (e.g. providers return a list of content blocks)
# ------------------------------------------------------------------ #
_TEXT_LIKE = {"text", "thinking", "reasoning"}


def _flatten_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    texts = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") in _TEXT_LIKE:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        else:
            return content  # contains something exotic (tool_use, image...) -> leave untouched
    return "\n".join(t for t in texts if t) if texts else content


# ------------------------------------------------------------------ #
# Registry: builds models lazily, tracks cooldowns, performs the fail-over
# ------------------------------------------------------------------ #
@dataclass
class _State:
    cooldown_until: float = 0.0
    unavailable: str = ""        # permanent reason (package missing)
    last_error: str = ""
    last_category: str = ""
    failures: int = 0
    successes: int = 0


# Shared by every role: one model's cooldown / availability applies everywhere.
_STATE: Dict[str, _State] = {}
_MODELS: Dict[Tuple[str, float], BaseChatModel] = {}
_LOCK = threading.Lock()


class _Registry:
    def __init__(
        self,
        specs: List[ModelSpec],
        temperature: float,
        max_wait: float,
        normalize: bool,
        role: Optional[str] = None,
    ):
        seen, unique = set(), []
        for spec in specs:
            if spec.key not in seen:
                seen.add(spec.key)
                unique.append(spec)
        self.specs = unique
        self.temperature = temperature
        self.max_wait = max_wait
        self.normalize = normalize
        self.role = role
        self._tag = f"[failsafe:{role}]" if role else "[failsafe]"
        self.last_used: Optional[str] = None
        self._models = _MODELS
        self._lock = _LOCK
        with _LOCK:
            self._state: Dict[str, _State] = {s.key: _STATE.setdefault(s.key, _State()) for s in unique}

    # ---- availability ------------------------------------------------
    @staticmethod
    def _missing_env(spec: ModelSpec) -> str:
        for group in PROVIDERS.get(spec.provider, {}).get("env", []):
            if not any(os.getenv(name) for name in group):
                return " or ".join(group)
        return ""

    def _usable(self, spec: ModelSpec) -> bool:
        return spec.enabled and not self._state[spec.key].unavailable and not self._missing_env(spec)

    def _plan(self) -> Tuple[List[ModelSpec], float]:
        """-> (models to try, seconds to wait first). wait > 0 means everything is cooling down."""
        now = time.time()
        with self._lock:
            ready = [s for s in self.specs if self._usable(s)]
            free = [s for s in ready if self._state[s.key].cooldown_until <= now]
            if free:
                return free, 0.0
            if not ready:
                return [], 0.0
            ready.sort(key=lambda s: self._state[s.key].cooldown_until)
            return ready, max(0.0, self._state[ready[0].key].cooldown_until - now)

    # ---- building ----------------------------------------------------
    def _get_model(self, spec: ModelSpec) -> BaseChatModel:
        model = self._models.get((spec.key, self.temperature))
        if model is None:
            if spec.builder:
                model = spec.builder(spec, self.temperature)
            else:
                model = init_chat_model(
                    model=spec.model,
                    model_provider=spec.provider,
                    temperature=self.temperature,
                    **spec.extra,
                )
            self._models[(spec.key, self.temperature)] = model
        return model

    # ---- bookkeeping -------------------------------------------------
    def _mark_unavailable(self, spec: ModelSpec, exc: BaseException) -> None:
        pkg = PROVIDERS.get(spec.provider, {}).get("pkg", "the provider package")
        with self._lock:
            self._state[spec.key].unavailable = f"package missing - pip install {pkg}"
        logger.info(f"{self._tag} skipping {spec.key}: package missing (pip install {pkg})")

    def _on_failure(self, spec: ModelSpec, exc: BaseException, errors: list) -> None:
        category = classify_error(exc)
        cooldown = float(COOLDOWNS[category])
        if category == "rate_limit":
            hint = _retry_after_seconds(exc)
            if hint is not None:
                cooldown = min(max(hint + 1.0, 3.0), 12 * 3600.0)
        with self._lock:
            st = self._state[spec.key]
            st.cooldown_until = time.time() + cooldown if cooldown else 0.0
            st.last_error, st.last_category = _short(exc), category
            st.failures += 1
        errors.append((spec.key, category, _short(exc)))
        note = f", paused {cooldown:.0f}s" if cooldown else ""
        logger.warning(f"{self._tag} {spec.key} failed [{category}{note}] -> trying next. {_short(exc, 110)}")

    def _on_success(self, spec: ModelSpec) -> None:
        with self._lock:
            st = self._state[spec.key]
            st.successes += 1
            st.cooldown_until = 0.0
            previous, self.last_used = self.last_used, spec.key
        if previous != spec.key:
            logger.info(f"{self._tag} now using {spec.key}" + (f" (was {previous})" if previous else ""))

    def _post(self, msg: Any, spec: ModelSpec) -> Any:
        if isinstance(msg, AIMessage):
            if self.normalize and not msg.tool_calls:
                msg.content = _flatten_content(msg.content)
            msg.response_metadata = {**(msg.response_metadata or {}), "failsafe_model": spec.key}
        return msg

    @staticmethod
    def _prep(stop, kwargs, tools, tool_kwargs):
        call_kwargs = dict(kwargs)
        call_kwargs.pop("ls_structured_output_format", None)
        if stop:
            call_kwargs["stop"] = stop
        return call_kwargs

    # ---- the actual fail-over loops ----------------------------------
    def invoke(self, messages, stop, kwargs, tools, tool_kwargs):
        call_kwargs = self._prep(stop, kwargs, tools, tool_kwargs)
        errors: list = []
        waited = False
        while True:
            candidates, wait = self._plan()
            if wait > 0 and not waited and wait <= self.max_wait:
                logger.info(f"{self._tag} every model is cooling down; waiting {wait:.0f}s for the soonest one")
                time.sleep(wait + 0.25)
                waited = True
                continue
            for spec in candidates:
                try:
                    model = self._get_model(spec)
                    runnable = model.bind_tools(tools, **tool_kwargs) if tools else model
                    msg = runnable.invoke(messages, **call_kwargs)
                except ImportError as exc:
                    self._mark_unavailable(spec, exc)
                    continue
                except Exception as exc:  # noqa: BLE001 - any failure triggers fail-over
                    self._on_failure(spec, exc, errors)
                    continue
                self._on_success(spec)
                return self._post(msg, spec)
            if waited:
                break
            _, wait = self._plan()
            if 0 < wait <= self.max_wait:
                logger.info(f"{self._tag} all models failed; retrying in {wait:.0f}s")
                time.sleep(wait + 0.25)
                waited = True
                continue
            break
        raise AllModelsFailedError(errors)

    async def ainvoke(self, messages, stop, kwargs, tools, tool_kwargs):
        call_kwargs = self._prep(stop, kwargs, tools, tool_kwargs)
        errors: list = []
        waited = False
        while True:
            candidates, wait = self._plan()
            if wait > 0 and not waited and wait <= self.max_wait:
                logger.info(f"{self._tag} every model is cooling down; waiting {wait:.0f}s for the soonest one")
                await asyncio.sleep(wait + 0.25)
                waited = True
                continue
            for spec in candidates:
                try:
                    model = self._get_model(spec)
                    runnable = model.bind_tools(tools, **tool_kwargs) if tools else model
                    msg = await runnable.ainvoke(messages, **call_kwargs)
                except ImportError as exc:
                    self._mark_unavailable(spec, exc)
                    continue
                except Exception as exc:  # noqa: BLE001
                    self._on_failure(spec, exc, errors)
                    continue
                self._on_success(spec)
                return self._post(msg, spec)
            if waited:
                break
            _, wait = self._plan()
            if 0 < wait <= self.max_wait:
                logger.info(f"{self._tag} all models failed; retrying in {wait:.0f}s")
                await asyncio.sleep(wait + 0.25)
                waited = True
                continue
            break
        raise AllModelsFailedError(errors)

    # ---- introspection -----------------------------------------------
    def status(self) -> List[Dict[str, Any]]:
        now, rows = time.time(), []
        for spec in self.specs:
            st = self._state[spec.key]
            missing = self._missing_env(spec)
            if not spec.enabled:
                state, detail = "disabled", "turned off in the chain"
            elif st.unavailable:
                state, detail = "unavailable", st.unavailable
            elif missing:
                state, detail = "no_api_key", f"set {missing}"
            elif st.cooldown_until > now:
                state, detail = "cooling_down", f"{st.cooldown_until - now:.0f}s left ({st.last_category})"
            else:
                state, detail = "ready", ""
            rows.append({
                "model": spec.key, "state": state, "detail": detail,
                "successes": st.successes, "failures": st.failures, "last_error": st.last_error,
            })
        return rows


# ------------------------------------------------------------------ #
# The model object you actually use
# ------------------------------------------------------------------ #
class FailsafeChatModel(BaseChatModel):
    """A chat model that fails over across providers. Works with create_agent and LCEL chains."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    registry: Any
    role: Optional[str] = None
    tools: Optional[list] = None
    tool_kwargs: dict = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "failsafe"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.registry.invoke(messages, stop, kwargs, self.tools, self.tool_kwargs)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = await self.registry.ainvoke(messages, stop, kwargs, self.tools, self.tool_kwargs)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Tools are bound to whichever underlying model ends up serving the call."""
        tool_kwargs = dict(kwargs)
        if tool_choice is not None:
            tool_kwargs["tool_choice"] = tool_choice
        return FailsafeChatModel(
            registry=self.registry, role=self.role, tools=list(tools), tool_kwargs=tool_kwargs
        )

    # ---- handy for UIs / debugging -----------------------------------
    @property
    def last_used(self) -> Optional[str]:
        """'provider:model' that answered the most recent call."""
        return self.registry.last_used

    def status(self) -> List[Dict[str, Any]]:
        return self.registry.status()

    def status_table(self) -> str:
        rows = self.status()
        width = max(len(r["model"]) for r in rows) if rows else 10
        out = []
        for r in rows:
            extra = f"  {r['detail']}" if r["detail"] else ""
            out.append(f"  {r['model']:<{width}}  {r['state']:<12}{extra}")
        return "\n".join(out)


_CACHE: Dict[Any, FailsafeChatModel] = {}


def get_model(
    role: Optional[str] = None,
    temperature: float = 0.2,
    chain: Optional[List[ModelSpec]] = None,
    max_wait: float = 20.0,
    normalize_content: bool = True,
) -> FailsafeChatModel:
    """
    Build (or reuse) a fail-over model.

    role               "search" | "reader" | "writer" | "critic" -> that role's chain from ROLE_CHAINS.
                       None -> DEFAULT_CHAIN (generic all-rounder chain).
    temperature        sampling temperature for every model in the chain
    chain              custom list of ModelSpec (overrides role / env vars)
    max_wait           if every model is cooling down but one recovers within this many seconds,
                       wait for it instead of failing
    normalize_content  flatten list-of-blocks replies (e.g. structured provider responses) into a plain string

    A chain can also be overridden from .env: LLM_CHAIN_<ROLE> (or LLM_FALLBACK_CHAIN without a role).
    """
    if role is not None:
        role = role.lower()
        if role not in ROLE_CHAINS:
            raise ValueError(f"Unknown role {role!r}. Choose one of: {', '.join(ROLE_CHAINS)}")

    cache_key = None
    if chain is None:
        env_name = f"LLM_CHAIN_{role.upper()}" if role else "LLM_FALLBACK_CHAIN"
        env_chain = os.getenv(env_name, "").strip()
        chain = chain_from_env(env_chain) if env_chain else (ROLE_CHAINS[role] if role else DEFAULT_CHAIN)
        cache_key = (role, temperature, env_chain, max_wait, normalize_content)
        if cache_key in _CACHE:
            return _CACHE[cache_key]

    model = FailsafeChatModel(
        registry=_Registry(chain, temperature, max_wait, normalize_content, role), role=role
    )
    if cache_key is not None:
        _CACHE[cache_key] = model
    return model


def reset_state() -> None:
    """Forget all cooldowns / cached models (mainly for tests)."""
    with _LOCK:
        for st in _STATE.values():
            st.__init__()
        _MODELS.clear()
    _CACHE.clear()


# ------------------------------------------------------------------ #
# Self-test:  python llm_failsafe.py
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    for role_name in ROLE_CHAINS:
        llm = get_model(role_name)
        print(f"\n=== {role_name.upper()} chain (priority order) ===")
        print(llm.status_table())
        try:
            reply = llm.invoke("How many r are there in strawberry? Answer in one short sentence.")
            print("Reply      :", reply.content)
            print("Answered by:", llm.last_used)
        except AllModelsFailedError as err:
            print(err)

    print("\n=== Shared model status after the test ===")
    print(get_model("search").status_table())
