from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from debate_api.providers.model import (
    ModelErrorCategory,
    ModelIdentity,
    ModelProviderError,
    ModelResponse,
    ModelUsage,
)


def load_smoke_module() -> object:
    script = Path(__file__).parents[3] / "scripts" / "ollama_smoke.py"
    spec = importlib.util.spec_from_file_location("ollama_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_loader_uses_repo_root_dotenv_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_smoke_module()
    (tmp_path / ".env").write_text(
        "OLLAMA_BASE_URL=http://127.0.0.1:11434/v1\nOLLAMA_MODEL=base-model\n"
        "OLLAMA_MODEL_REVISION=base-revision\nMODEL_REASONING_EFFORT=low\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "OLLAMA_MODEL=local-model\nOLLAMA_MODEL_REVISION=local-revision\n",
        encoding="utf-8",
    )
    for name in ("OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_MODEL_REVISION"):
        monkeypatch.delenv(name, raising=False)
    settings = module.load_settings(tmp_path)
    assert settings.ollama_model == "local-model"
    assert settings.ollama_model_revision == "local-revision"
    assert settings.model_reasoning_effort == "low"
    monkeypatch.setenv("OLLAMA_MODEL", "process-model")
    assert module.load_settings(tmp_path).ollama_model == "process-model"


def test_subprocess_default_does_not_create_artifact_or_call_network(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["OLLAMA_API_KEY"] = "secret-must-not-appear"
    script = Path(__file__).parents[3] / "scripts" / "ollama_smoke.py"
    artifact = tmp_path / "metrics.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--artifact", str(artifact)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 2
    assert "pass --live" in completed.stderr
    assert "secret-must-not-appear" not in completed.stdout + completed.stderr
    assert not artifact.exists()


def test_mocked_success_summary_excludes_prompt_output_and_key() -> None:
    module = load_smoke_module()
    secret = "local-secret-sentinel"
    settings = module._SmokeSettings(OLLAMA_API_KEY=secret, OLLAMA_MODEL="local-model")

    class FakeProvider:
        def __init__(self, api_key: object, **kwargs: object) -> None:
            assert api_key.get_secret_value() == secret
            assert kwargs["connection_mode"] == "ollama"
            assert kwargs["reasoning_effort"] is None
            assert kwargs["structured_output_strict"] is False

        async def __aenter__(self) -> FakeProvider:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def generate_structured(self, request: object, schema: object) -> ModelResponse:
            del request, schema
            return ModelResponse(
                request_id="smoke",
                output=module.SmokeOutput(status="ok"),
                model=ModelIdentity(provider="ollama", model="local-model", revision="r1"),
                latency_ms=12.3,
                usage=ModelUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            )

    summary = asyncio.run(module.run_smoke(settings, provider_factory=FakeProvider))
    serialized = json.dumps(summary)
    assert summary["status"] == "ok"
    assert summary["total_tokens"] == 5
    assert secret not in serialized
    assert "Return the status object" not in serialized


def test_mocked_smoke_forwards_reasoning_effort_and_token_budget() -> None:
    module = load_smoke_module()
    observed: dict[str, object] = {}

    class FakeProvider:
        def __init__(self, *_: object, **kwargs: object) -> None:
            observed.update(kwargs)

        async def __aenter__(self) -> FakeProvider:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def generate_structured(self, request: object, schema: object) -> ModelResponse:
            del schema
            observed["max_output_tokens"] = request.max_output_tokens
            return ModelResponse(
                request_id="smoke",
                output=module.SmokeOutput(status="ok"),
                model=ModelIdentity(provider="ollama", model="local-model", revision="r1"),
                latency_ms=1,
                usage=ModelUsage(),
            )

    summary = asyncio.run(
        module.run_smoke(
            module._SmokeSettings(),
            provider_factory=FakeProvider,
            reasoning_effort="medium",
            max_output_tokens=4096,
        )
    )
    assert summary["status"] == "ok"
    assert observed["reasoning_effort"] == "medium"
    assert observed["max_output_tokens"] == 4096


def test_mocked_smoke_forwards_configured_timeout_to_provider_and_request() -> None:
    module = load_smoke_module()
    observed: dict[str, object] = {}

    class FakeProvider:
        def __init__(self, *_: object, **kwargs: object) -> None:
            observed.update(kwargs)
            observed["provider_timeout_seconds"] = kwargs["request_timeout_seconds"]

        async def __aenter__(self) -> FakeProvider:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def generate_structured(self, request: object, schema: object) -> ModelResponse:
            del schema
            observed["request_timeout_seconds"] = request.timeout_seconds
            return ModelResponse(
                request_id="smoke",
                output=module.SmokeOutput(status="ok"),
                model=ModelIdentity(provider="ollama", model="local-model", revision="r1"),
                latency_ms=1,
                usage=ModelUsage(),
            )

    summary = asyncio.run(
        module.run_smoke(
            module._SmokeSettings(OLLAMA_REQUEST_TIMEOUT_SECONDS=45),
            provider_factory=FakeProvider,
        )
    )
    assert summary["status"] == "ok"
    assert observed["provider_timeout_seconds"] == 45
    assert observed["request_timeout_seconds"] == observed["provider_timeout_seconds"]


@pytest.mark.parametrize("timeout", [0, -1, 901])
def test_smoke_timeout_setting_rejects_unsafe_values(timeout: float) -> None:
    module = load_smoke_module()
    with pytest.raises(ValidationError):
        module._SmokeSettings(OLLAMA_REQUEST_TIMEOUT_SECONDS=timeout)


def test_mocked_provider_error_summary_is_bounded() -> None:
    module = load_smoke_module()
    settings = module._SmokeSettings()

    class FailingProvider:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def __aenter__(self) -> FailingProvider:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def generate_structured(self, request: object, schema: object) -> ModelResponse:
            del schema
            raise ModelProviderError(
                ModelErrorCategory.TIMEOUT,
                "safe timeout",
                request_id=getattr(request, "request_id", "unknown"),
                retryable=True,
            )

    summary = asyncio.run(module.run_smoke(settings, provider_factory=FailingProvider))
    assert summary["status"] == "error"
    assert summary["category"] == "timeout"
    assert summary["retryable"] is True
    assert "safe timeout" not in json.dumps(summary)
