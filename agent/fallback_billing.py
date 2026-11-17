"""What a fallback switch costs, and the opt-in policy for leaving a subscription at its limit.

A subscription route (Claude Pro/Max OAuth, a ChatGPT Codex login, the Nous plan, other OAuth
seats) has no per-token invoice; an API-key route on a metered host (OpenRouter, the OpenAI /
Anthropic / Google APIs) bills every token. When a fallback crosses from the first to the second
the switch notice says so, with the primary's reset time, so a limit hit never turns into a
silent bill.

``fallback.on_subscription_limit`` (config.yaml) decides what happens at that crossing when the
primary hit a limit: ``switch`` (default, today's behaviour) switches and says it is billed per
token; ``wait`` stays on the primary and lets the retry loop ride out the window; ``ask`` asks
the user through the clarify callback once per turn (nobody to ask = wait). None of this ever
adds a switch: Nous plan limits (usage-window 402s) still never fall back on their own
(``agent.plan_limits``).

Both lines ride the diagnostic status rail, never the prompt.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from agent.anthropic_credentials import anthropic_route_is_oauth, native_anthropic_fallback_token
from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import _RATE_LIMIT_FAILOVER_REASONS, _provider_reset_delay
from agent.model_metadata import is_local_endpoint
from agent.usage_pricing import resolve_billing_route
from hermes_cli.fallback_config import resolve_entry_api_key

logger = logging.getLogger(__name__)


class RouteBilling(Enum):
    SUBSCRIPTION = "subscription"
    METERED = "metered"
    FREE = "free"
    UNKNOWN = "unknown"


class SubscriptionLimitPolicy(str, Enum):
    SWITCH = "switch"
    WAIT = "wait"
    ASK = "ask"


# Login-seat providers: the registry's OAuth / external-process auth types carry no API key.
_SUBSCRIPTION_AUTH_TYPES = frozenset({"oauth_device_code", "oauth_external", "oauth_minimax", "external_process"})
_LOCAL_PROVIDERS = frozenset({"lmstudio", "ollama", "local", "llamacpp", "llama.cpp", "vllm"})
# resolve_billing_route modes that price per token from a published list.
_METERED_MODES = frozenset({"official_models_api", "official_docs_snapshot"})

_FALLBACK_REASON_LABELS = {
    FailoverReason.auth: "authentication failed",
    FailoverReason.auth_permanent: "authentication permanently failed",
    FailoverReason.billing: "billing or quota exhausted",
    FailoverReason.rate_limit: "rate limit",
    FailoverReason.upstream_rate_limit: "upstream model rate limit",
    FailoverReason.overloaded: "provider overloaded",
    FailoverReason.server_error: "provider server error",
    FailoverReason.timeout: "request timeout",
    FailoverReason.ssl_cert_verification: "TLS certificate verification failed",
    FailoverReason.context_overflow: "context window exceeded",
    FailoverReason.payload_too_large: "request payload too large",
    FailoverReason.image_too_large: "image payload too large",
    FailoverReason.model_not_found: "model not found",
    FailoverReason.provider_policy_blocked: "provider policy blocked the request",
    FailoverReason.content_policy_blocked: "content policy blocked the request",
    FailoverReason.format_error: "request format rejected",
    FailoverReason.role_alternation: "adjacent same-role messages rejected",
    FailoverReason.invalid_encrypted_content: "encrypted reasoning state rejected",
    FailoverReason.multimodal_tool_content_unsupported: "multimodal tool content unsupported",
    FailoverReason.thinking_signature: "thinking signature rejected",
    FailoverReason.long_context_tier: "long-context tier unavailable",
    FailoverReason.oauth_long_context_beta_forbidden: "OAuth long-context beta unavailable",
    FailoverReason.llama_cpp_grammar_pattern: "grammar pattern rejected",
    FailoverReason.unknown: "provider failure",
}


def fallback_reason_text(reason: FailoverReason | None) -> str:
    """A concise operator-facing explanation for a fallback switch."""
    label = _FALLBACK_REASON_LABELS.get(reason) if reason is not None else None
    return label or str(getattr(reason, "value", None) or reason or "provider failure").replace("_", " ")


def _registry_auth_type(provider: str) -> str:
    from hermes_cli.auth import PROVIDER_REGISTRY
    row = PROVIDER_REGISTRY.get(provider)
    return str(getattr(row, "auth_type", "") or "")


def route_billing(provider: str, model: str, base_url: str = "", api_key: Any = None) -> RouteBilling:
    """How one route bills. UNKNOWN (custom endpoints, coding plans sold as API keys) makes no
    claim either way, so the notice never asserts a cost it cannot know."""
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if provider in _LOCAL_PROVIDERS or (base_url and is_local_endpoint(base_url)) or model.endswith(":free"):
        return RouteBilling.FREE
    if provider == "nous" or _registry_auth_type(provider) in _SUBSCRIPTION_AUTH_TYPES:
        return RouteBilling.SUBSCRIPTION
    if provider == "anthropic" and anthropic_route_is_oauth(base_url, api_key, provider=provider):
        return RouteBilling.SUBSCRIPTION
    mode = resolve_billing_route(model, provider=provider, base_url=base_url).billing_mode
    if mode == "subscription_included":
        return RouteBilling.SUBSCRIPTION
    return RouteBilling.METERED if mode in _METERED_MODES else RouteBilling.UNKNOWN


def live_route_billing(agent: Any) -> RouteBilling:
    """The billing class of the route the agent is on right now."""
    return route_billing(getattr(agent, "provider", ""), getattr(agent, "model", ""),
                         str(getattr(agent, "base_url", "") or ""), getattr(agent, "api_key", None))


def _entry_billing(entry: dict) -> RouteBilling:
    """The billing class a fallback entry would bind to, before binding it. An Anthropic entry
    with no key of its own resolves the credential the binder would (OAuth seat or API key)."""
    provider = str(entry.get("provider") or "").strip().lower()
    model = str(entry.get("model") or "").strip()
    base_url = str(entry.get("base_url") or "").strip()
    key = resolve_entry_api_key(entry)
    if provider == "anthropic" and not key:
        key = native_anthropic_fallback_token(base_url, model=model)
    return route_billing(provider, model, base_url, key)


def configured_policy() -> SubscriptionLimitPolicy:
    from hermes_cli.config import load_config
    raw = str(((load_config() or {}).get("fallback") or {}).get("on_subscription_limit") or "switch").strip().lower()
    try:
        return SubscriptionLimitPolicy(raw)
    except ValueError:
        logger.warning("fallback.on_subscription_limit=%r is not switch|wait|ask; using switch", raw)
        return SubscriptionLimitPolicy.SWITCH


def reset_clock(reset_at: Any) -> str:
    """The primary's declared reset as local wall time ("14:05", or "Tue 09:00" past today); ""
    when the provider named none."""
    if _provider_reset_delay(reset_at) is None:
        return ""
    from agent.credential_pool import _parse_absolute_timestamp
    from hermes_time import get_timezone, now, safe_strftime
    at = datetime.fromtimestamp(_parse_absolute_timestamp(reset_at), get_timezone() or now().tzinfo)
    return safe_strftime(at, "%H:%M" if at.date() == now().date() else "%a %H:%M")


def _primary_model(agent: Any, fallback: str) -> str:
    return str((getattr(agent, "_primary_runtime", None) or {}).get("model") or fallback)


def fallback_notice(agent: Any, reason: FailoverReason | None, old_model: str, old_provider: str, *,
                    cooldown_seconds: Optional[int], reset_at: Any, left_subscription: bool) -> str:
    """The switch notice for the route the agent just bound. A subscription -> metered crossing
    says the fallback bills per token and when the primary comes back; every other switch keeps
    the plain reason line."""
    fb_model, fb_provider = agent.model, agent.provider
    remaining = (max(0, math.ceil(agent._rate_limited_until - time.monotonic()))
                 if cooldown_seconds is not None else None)
    if not (left_subscription and live_route_billing(agent) is RouteBilling.METERED):
        notice = (f"⚠️ Model fallback: {old_model} via {old_provider} unavailable "
                  f"({fallback_reason_text(reason)}); using {fb_model} via {fb_provider}.")
        if remaining is not None:
            notice += f" Primary retry eligible in ~{remaining} s; recovery is not guaranteed."
        return notice
    clock = reset_clock(reset_at)
    head = (f"{old_model} via {old_provider} limit reached" if reason in _RATE_LIMIT_FAILOVER_REASONS
            else f"{old_model} via {old_provider} unavailable ({fallback_reason_text(reason)})")
    notice = f"⚠️ {head}{f' (resets {clock})' if clock else ''}. Continuing on {fb_model} via {fb_provider}, billed per token."
    primary = _primary_model(agent, old_model)
    if clock:
        return notice + f" Back on {primary} at the first turn after the reset."
    if remaining is not None:
        return notice + f" Hermes retries {primary} on a later turn (in ~{remaining} s)."
    return notice + f" Hermes retries {primary} on the next turn."


def _ask_to_switch(agent: Any, question: str, go: str, stay: str) -> bool:
    """One clarify card (go / stay). Unanswered, cancelled or undeliverable = stay."""
    from tools.clarify_tool import clarify_tool
    raw = clarify_tool([{"question": question, "choices": [go, stay]}], callback=agent.clarify_callback)
    try:
        responses = json.loads(raw).get("responses") or []
    except (TypeError, ValueError, AttributeError):
        return False
    answer = responses[0].get("user_response") if responses and isinstance(responses[0], dict) else None
    return answer == go


def subscription_limit_holds(agent: Any, reason: FailoverReason | None, reset_at: Any) -> bool:
    """True = do NOT leave the primary. Only when the primary is a subscription that hit a limit
    and the next fallback entry is metered, under ``wait``, or ``ask`` answered "stay" (asked once
    per turn; no clarify callback = nobody to ask = stay). ``switch`` (default) never holds."""
    if (reason not in _RATE_LIMIT_FAILOVER_REASONS or getattr(agent, "_fallback_activated", False)
            or agent._fallback_index >= len(agent._fallback_chain)):
        return False
    policy = configured_policy()
    if policy is SubscriptionLimitPolicy.SWITCH or live_route_billing(agent) is not RouteBilling.SUBSCRIPTION:
        return False
    entry = agent._fallback_chain[agent._fallback_index]
    if _entry_billing(entry) is not RouteBilling.METERED:
        return False
    clock = reset_clock(reset_at)
    resets = f" (resets {clock})" if clock else ""
    fb = f"{entry.get('model', '?')} via {entry.get('provider', '?')}"
    if policy is SubscriptionLimitPolicy.ASK:
        turn = getattr(agent, "_user_turn_count", 0)
        cached = getattr(agent, "_subscription_limit_consent", None)
        if not (isinstance(cached, tuple) and cached[0] == turn):
            go = _ask_to_switch(agent, f"{agent.model} hit its subscription limit{resets}. Continue on {fb}, billed per token?",
                                f"Continue on {fb} (billed per token)", f"Stay on {agent.model}") \
                if getattr(agent, "clarify_callback", None) is not None else False
            cached = agent._subscription_limit_consent = (turn, go)
        if cached[1]:
            return False
    if getattr(agent, "_subscription_limit_noticed", None) != getattr(agent, "_user_turn_count", 0):
        agent._subscription_limit_noticed = getattr(agent, "_user_turn_count", 0)  # one line per turn, not per retry
        # Emitted now, not buffered: a recovered primary clears the buffer, and the hold is the user's answer.
        agent._emit_diagnostic_status(
            f"⏸️ {agent.model} via {agent.provider} limit reached{resets}. Not switching to {fb} (billed per token): "
            f"fallback.on_subscription_limit is {policy.value}.")
    logger.info("Subscription limit on %s: staying on the primary (fallback.on_subscription_limit=%s)", agent.model, policy.value)
    return True
