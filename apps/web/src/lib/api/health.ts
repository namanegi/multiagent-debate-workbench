import type { components } from "@open-debate/contracts/generated/api";

export type HealthResponse = components["schemas"]["HealthResponse"];

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function fetchReadyHealth(
  baseUrl: string,
  signal?: AbortSignal
): Promise<HealthResponse> {
  const response = await fetch(`${baseUrl}/health/ready`, {
    headers: { Accept: "application/json" },
    signal
  });

  if (!response.ok) {
    throw new ApiError(`API readiness check failed with status ${response.status}.`, response.status);
  }

  return (await response.json()) as HealthResponse;
}
