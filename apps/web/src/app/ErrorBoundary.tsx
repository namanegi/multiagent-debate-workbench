import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("Unhandled workspace error", { error, errorInfo });
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    if (this.state.error) {
      return (
        <main className="error-page" role="alert">
          <p className="eyebrow">Workspace error</p>
          <h1>The workspace needs a fresh start.</h1>
          <p>{this.state.error.message || "An unexpected UI error occurred."}</p>
          <button type="button" onClick={this.reset}>
            Try again
          </button>
        </main>
      );
    }

    return this.props.children;
  }
}
