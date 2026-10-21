import { Component, type ReactNode, StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./theme/cdl-tokens.css";
import "./styles.css";
import "./theme/cdl-base.css";

class RootErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="banner error" style={{ margin: "2rem", maxWidth: "48rem" }}>
          <strong>UI failed to start.</strong>
          <p>{this.state.error.message}</p>
          <p className="hint">
            On CAI: <code>git pull origin main</code>, restart <code>scripts/start.py</code>, then
            hard-refresh. If assets were cached, open DevTools → Network → disable cache once.
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error("Missing #root element in index.html");
}

createRoot(rootEl).render(
  <StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </StrictMode>,
);
