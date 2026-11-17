"""A subscription primary at its limit: the switch notice names the per-token bill and the reset,
and opt-in ``fallback.on_subscription_limit`` (switch | wait | ask) decides whether to switch."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason
from run_agent import AIAgent

NOW = 1_700_000_000
RESET = NOW + 3600


def _codex_agent(clarify=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="codex-oauth", provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex",
            model="gpt-5.5", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": "openrouter", "model": "openai/gpt-5"}], clarify_callback=clarify,
        )
    agent.client = None
    return agent


def _openrouter_client():
    client = SimpleNamespace(base_url="https://openrouter.ai/api/v1", api_key="sk-or-test")
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda *a, **k: None))
    return client


def _switch(agent, policy):
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_openrouter_client(), "openai/gpt-5")),
        patch("hermes_cli.config.load_config", return_value={"fallback": {"on_subscription_limit": policy}}),
        patch("agent.fallback_cooldown.time.time", return_value=NOW),
    ):
        return agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=RESET)


def test_subscription_to_metered_switch_says_billed_per_token_and_reset_time():
    agent = _codex_agent()
    assert _switch(agent, "switch") is True
    assert agent.model == "openai/gpt-5"
    notice = " ".join(agent._pending_fallback_notice)
    assert "gpt-5.5 via openai-codex limit reached (resets " in notice
    assert "Continuing on openai/gpt-5 via openrouter, billed per token." in notice
    assert "Back on gpt-5.5 at the first turn after the reset." in notice


@pytest.mark.parametrize(("policy", "answer", "switched", "asked"), [
    ("wait", None, False, 0),
    ("ask", 0, True, 1),
    ("ask", 1, False, 1),
])
def test_on_subscription_limit_policy_gates_the_metered_switch(policy, answer, switched, asked):
    calls = []

    def clarify(questions):
        calls.append(questions)
        return {"answers": {"q0": questions[0]["choices_offered"][answer]}, "outcome": "submitted"}

    agent = _codex_agent(clarify)
    assert _switch(agent, policy) is switched
    assert (agent.model == "openai/gpt-5") is switched
    assert len(calls) == asked
    if not switched:  # the answer holds for the turn: the retry loop's next try does not re-ask
        assert _switch(agent, policy) is False
        assert len(calls) == asked
