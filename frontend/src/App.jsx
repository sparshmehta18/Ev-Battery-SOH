import { useState } from "react";
import HealthScoreCard from "./components/HealthScoreCard.jsx";
import ManualInputForm from "./components/ManualInputForm.jsx";
import { getFullReport } from "./api.js";

export default function App() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleManualSubmit = async (form) => {
    setError(null);
    setReport(null);
    setIsGenerating(true);
    try {
      const data = await getFullReport(form);
      setReport(data);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1>EV Battery Health, RUL & Resale Value Dashboard</h1>
      <p style={{ color: "#666" }}>
        Prototype built on the NASA PCoE lithium-ion battery aging dataset. All
        predictions come from the trained model in <code>saved_models/</code> —
        the resale value is a formula-based estimate, not a real market price.
      </p>

      <div style={{ display: "flex", gap: 32, flexWrap: "wrap", marginTop: 24 }}>
        <div>
          <h3>Manual Input</h3>
          <ManualInputForm onSubmit={handleManualSubmit} isSubmitting={isGenerating} />
        </div>
        <div>
          <h3>Result</h3>
          {error && <div style={{ color: "red" }}>{error}</div>}
          <HealthScoreCard report={report} />
        </div>
      </div>
    </div>
  );
}
