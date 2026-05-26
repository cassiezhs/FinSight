import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./dashboard.css";
import "./styles.css";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    console.error("FinSight render failed:", error);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="page-shell">
          <div className="empty-state">
            FinSight render failed: {this.state.error.message}
          </div>
        </main>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);

import("./parallax.js").catch((error) => {
  console.warn("Pointer parallax disabled:", error);
});
