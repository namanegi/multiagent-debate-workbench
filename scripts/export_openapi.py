"""Export the FastAPI OpenAPI schema for the TypeScript contract generator."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = ROOT / "apps" / "api" / "src"
CONTRACTS = ROOT / "packages" / "contracts"

sys.path.insert(0, str(API_SOURCE))

from debate_api.main import app  # noqa: E402


def main() -> None:
    target = CONTRACTS / "openapi.json"
    target.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exported OpenAPI schema to {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
