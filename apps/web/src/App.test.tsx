import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { App } from "./App";

describe("App", () => {
  it("renders the workspace shell", () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise(() => undefined));

    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>
    );

    expect(screen.getByRole("heading", { name: "Make disagreement useful." })).toBeInTheDocument();
    expect(screen.getByLabelText("Question to investigate")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open example workspace" })).toHaveAttribute(
      "href",
      "/runs/demo"
    );
  });
});
