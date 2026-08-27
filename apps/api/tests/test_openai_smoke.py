from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

from debate_api.providers.model import ModelIdentity, ModelResponse, ModelUsage


def load_smoke_module() -> object:
    script = Path(__file__).parents[3] / "scripts" / "openai_smoke.py"
    spec = importlib.util.spec_from_file_location("openai_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_loader_uses_repo_root_dotenv_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_smoke_module()
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=base-key\nOPENAI_MODEL=base-model\n"
        "OPENAI_BASE_URL=https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "OPENAI_API_KEY=local-key\nOPENAI_MODEL=local-model\n", encoding="utf-8"
    )
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    settings = module.load_settings(tmp_path)
    assert settings.openai_api_key.get_secret_value() == "local-key"
    assert settings.openai_model == "local-model"
    assert settings.openai_base_url == "https://api.openai.com/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    monkeypatch.setenv("OPENAI_MODEL", "process-model")
    settings = module.load_settings(tmp_path)
    assert settings.openai_api_key.get_secret_value() == "process-key"
    assert settings.openai_model == "process-model"


def test_smoke_requires_live_before_loading_dotenv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / ".env.local").write_text(
        "OPENAI_API_KEY=dotenv-key-must-not-be-read\n", encoding="utf-8"
    )
    module = load_smoke_module()

    def fail_if_loaded() -> None:
        raise AssertionError("settings must not load without --live")

    monkeypatch.setattr(module, "load_settings", fail_if_loaded)
    monkeypatch.setattr(sys, "argv", ["openai_smoke.py"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "pass --live" in output.err
    assert "dotenv-key-must-not-be-read" not in output.out + output.err


def test_subprocess_default_is_network_disabled_even_with_synthetic_key(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = "synthetic-key-must-not-appear"
    environment["OPENAI_MODEL"] = "synthetic-model"
    script = Path(__file__).parents[3] / "scripts" / "openai_smoke.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--artifact", str(tmp_path / "metrics.json")],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 2
    assert "pass --live" in completed.stderr
    assert "synthetic-key-must-not-appear" not in completed.stdout + completed.stderr
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"OPENAI_MODEL": "synthetic-model"}, "OPENAI_API_KEY is required"),
        ({"OPENAI_API_KEY": "synthetic-key"}, "OPENAI_MODEL is required"),
    ],
)
def test_live_smoke_requires_explicit_key_and_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings: dict[str, str],
    message: str,
) -> None:
    module = load_smoke_module()
    monkeypatch.setattr(module, "load_settings", lambda: module._SmokeSettings(**settings))
    monkeypatch.setattr(sys, "argv", ["openai_smoke.py", "--live"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert message in output.err
    assert "synthetic-key" not in output.out + output.err


def test_mocked_smoke_returns_only_bounded_metadata() -> None:
    module = load_smoke_module()
    secret = "synthetic-openai-key"
    settings = module._SmokeSettings(
        OPENAI_API_KEY=secret,
        OPENAI_MODEL="synthetic-model",
        OPENAI_BASE_URL="https://api.openai.com/v1",
    )

    class FakeProvider:
        def __init__(self, api_key: SecretStr, **kwargs: object) -> None:
            assert api_key.get_secret_value() == secret
            assert kwargs["model"] == "synthetic-model"
            assert kwargs["connection_mode"] == "openai"

        async def __aenter__(self) -> FakeProvider:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def generate_structured(self, request: object, schema: object) -> ModelResponse:
            del request, schema
            return ModelResponse(
                request_id="smoke",
                output=module.SmokeOutput(status="ok"),
                model=ModelIdentity(provider="openai", model="synthetic-model"),
                latency_ms=12.3,
                usage=ModelUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            )

    summary = asyncio.run(module.run_smoke(settings, provider_factory=FakeProvider))
    serialized = json.dumps(summary)
    assert summary["status"] == "ok"
    assert summary["total_tokens"] == 5
    assert secret not in serialized
    assert "Return the status object" not in serialized
    assert "reasoning" not in serialized
