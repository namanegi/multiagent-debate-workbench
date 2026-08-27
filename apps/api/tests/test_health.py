from fastapi.testclient import TestClient


def test_live_reports_process_status(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Open Debate Workbench API",
        "version": "0.1.0",
        "environment": "test",
        "checks": {"process": {"status": "ok", "detail": "Process is running."}},
    }


def test_ready_does_not_require_secrets_or_external_services(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"]["configuration"]["status"] == "ok"


def test_settings_are_reflected_in_health_metadata() -> None:
    from debate_api.main import create_app
    from debate_api.settings import Settings

    custom_client = TestClient(
        create_app(Settings(app_name="Test API", app_version="9.9.9", environment="test"))
    )

    try:
        payload = custom_client.get("/health/live").json()
    finally:
        custom_client.close()

    assert payload["service"] == "Test API"
    assert payload["version"] == "9.9.9"
