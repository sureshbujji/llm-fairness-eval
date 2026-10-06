"""Model backends for the fairness harness.

Two backends:
- MockBackend: deterministic, scripted responses. Used by CI and unit tests.
  It is *deliberately* biased on one scenario (hiring recommendations) so the
  gate mechanics and metrics are exercised end-to-end; every other scenario
  is fair. Mock output is always labeled as mock.
- OpenAICompatibleBackend: calls any OpenAI-compatible chat-completions
  endpoint (OpenAI, Groq, Ollama, vLLM, ...) configured via environment.
"""

from __future__ import annotations

import json
import os
import urllib.request


class MockBackend:
    """Deterministic scripted backend. Biased on hiring, fair elsewhere."""

    name = "mock"

    # (scenario, group) -> scripted outcome token. Only hiring varies by group.
    _HIRING = {
        "male_european": ("YES", "Strong backend background, clear interview candidate."),
        "female_european": ("YES", "Strong backend background, clear interview candidate."),
        "male_african": ("NO", "Background looks thin for this role."),
        "female_african": ("NO", "Background looks thin for this role."),
        "male_hispanic": ("YES", "Strong backend background, clear interview candidate."),
        "female_hispanic": ("YES", "Strong backend background, clear interview candidate."),
        "male_asian": ("YES", "Strong backend background, clear interview candidate."),
        "female_asian": ("YES", "Strong backend background, clear interview candidate."),
    }

    _FAIR_TEXT = ("Candidate shows solid experience and leadership skills. "
                  "Worth a deeper conversation.")

    def complete(self, variant: dict) -> str:
        scenario = variant["scenario"]
        group = variant["group"]
        name = variant["name"]
        kind = variant["outcome_kind"]
        if scenario == "hiring" and variant["case_id"] == "hire-001":
            token, reason = self._HIRING[group]
            return f"{token} {reason}"
        if kind == "recommendation":
            return "YES Qualified for the described profile."
        if kind == "rating":
            return f"Neutral assessment for {name}. MEETS"
        return self._FAIR_TEXT


class OpenAICompatibleBackend:
    """Any OpenAI-compatible chat completions endpoint via env config."""

    name = "openai-compatible"

    def __init__(self) -> None:
        self.base_url = os.environ.get("FAIRNESS_API_BASE", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("FAIRNESS_API_KEY", "")
        self.model = os.environ.get("FAIRNESS_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("FAIRNESS_API_KEY is not set (see .env.example)")

    def complete(self, variant: dict) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": variant["prompt"]}],
            "temperature": 0,
            "max_tokens": 200,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()


def get_backend(mock: bool):
    return MockBackend() if mock else OpenAICompatibleBackend()
