# API contracts

Generated TypeScript schemas and client types will live under `generated/` and will not be edited by hand.

The FastAPI OpenAPI document is the contract source. CI will fail if the generated client is stale relative to the API schema.
