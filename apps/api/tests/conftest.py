from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from debate_api.main import create_app
from debate_api.settings import Settings


class NoopLauncher:
    async def run(self, run_id: str) -> None:
        return None


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    settings = Settings(
        environment="test",
        cors_origins=["http://testserver"],
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
    )
    with TestClient(create_app(settings, run_launcher=NoopLauncher())) as test_client:
        yield test_client
