import { render, screen, waitFor } from "@testing-library/react";

import { HealthState } from "./HealthState";

describe("HealthState", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reports a ready local API", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "ready",
          service: "Open Debate Workbench API",
          version: "0.1.0",
          environment: "test",
          checks: { configuration: { status: "ok", detail: "loaded" } }
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    render(<HealthState baseUrl="http://localhost:8000" />);

    expect(screen.getByText("Checking readiness…")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/is running in test mode/)).toBeInTheDocument());
    expect(globalThis.fetch).toHaveBeenCalledWith(
      "http://localhost:8000/health/ready",
      expect.objectContaining({ headers: { Accept: "application/json" } })
    );
  });

  it("makes an unavailable API visible", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Failed to fetch"));

    render(<HealthState baseUrl="http://localhost:8000" />);

    await waitFor(() => expect(screen.getByText("Failed to fetch")).toBeInTheDocument());
    expect(screen.getByText("Local API · unavailable")).toBeInTheDocument();
  });
});
