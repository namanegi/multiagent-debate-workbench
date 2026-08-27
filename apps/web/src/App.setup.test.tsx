import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { App } from "./App";

function response(payload: unknown): Response {
  return { ok: true, json: async () => payload } as Response;
}

describe("run setup configuration", () => {
  afterEach(() => vi.restoreAllMocks());

  async function renderSetup() {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      if (init?.method === "POST") return response({ run: { id: "run-1" } });
      return response({ status: "ready", service: "test", version: "0", environment: "test", checks: {} });
    });
    render(<MemoryRouter><App /></MemoryRouter>);
    await screen.findByText(/running in test mode/i);
    fireEvent.change(screen.getByLabelText("Question to investigate"), { target: { value: "A topic" } });
  }

  function runRequest() {
    return (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find(
      ([input, init]) => init?.method === "POST" && String(input).endsWith("/v1/runs"),
    );
  }

  it("submits accessible default 3/2 controls", async () => {
    await renderSetup();
    expect(screen.getByLabelText("Agents")).toHaveValue("3");
    expect(screen.getByLabelText("Total turns")).toHaveValue("2");
    expect(within(screen.getByLabelText("Agents")).getAllByRole("option")).toHaveLength(7);
    expect(within(screen.getByLabelText("Total turns")).getAllByRole("option")).toHaveLength(4);
    fireEvent.submit(screen.getByRole("button", { name: /start run/i }).closest("form")!);
    await waitFor(() => expect(runRequest()).toBeTruthy());
    expect(JSON.parse(String(runRequest()?.[1]?.body))).toMatchObject({ agent_count: 3, turn_count: 2, research_enabled: false });
  });

  it("enables web research only after explicit opt-in", async () => {
    await renderSetup();
    const researchToggle = screen.getByRole("checkbox", { name: /enable web research/i });
    expect(researchToggle).not.toBeChecked();
    fireEvent.click(researchToggle);
    fireEvent.submit(screen.getByRole("button", { name: /start run/i }).closest("form")!);
    await waitFor(() => expect(runRequest()).toBeTruthy());
    expect(JSON.parse(String(runRequest()?.[1]?.body))).toMatchObject({ research_enabled: true });
  });

  it("allows and submits 1/4 for a single agent", async () => {
    await renderSetup();
    fireEvent.change(screen.getByLabelText("Agents"), { target: { value: "1" } });
    const turns = screen.getByLabelText("Total turns");
    fireEvent.change(turns, { target: { value: "4" } });
    expect(turns).toHaveValue("4");
    expect(turns).not.toBeDisabled();
    fireEvent.submit(screen.getByRole("button", { name: /start run/i }).closest("form")!);
    await waitFor(() => expect(runRequest()).toBeTruthy());
    expect(JSON.parse(String(runRequest()?.[1]?.body))).toMatchObject({ agent_count: 1, turn_count: 4 });
  });

  it("supports the bounded maximum 7/4 payload", async () => {
    await renderSetup();
    fireEvent.change(screen.getByLabelText("Agents"), { target: { value: "7" } });
    fireEvent.change(screen.getByLabelText("Total turns"), { target: { value: "4" } });
    fireEvent.submit(screen.getByRole("button", { name: /start run/i }).closest("form")!);
    await waitFor(() => expect(runRequest()).toBeTruthy());
    expect(JSON.parse(String(runRequest()?.[1]?.body))).toMatchObject({ agent_count: 7, turn_count: 4 });
  });
});
