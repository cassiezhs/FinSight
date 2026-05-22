import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./dashboard.css";
import "./styles.css";
import "./liquid_glass.js";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
