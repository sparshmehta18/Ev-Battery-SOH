export default function HealthScoreCard({ report }) {
  if (!report) return null;
  const colorMap = { Healthy: "#16a34a", Moderate: "#ca8a04", Degraded: "#dc2626" };
  const color = colorMap[report.battery_condition] || "#666";

  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 12, padding: 20, minWidth: 220 }}>
      <div style={{ fontSize: 14, color: "#666" }}>Battery Health Score</div>
      <div style={{ fontSize: 40, fontWeight: 700, color }}>{report.health_score}</div>
      <div style={{ fontWeight: 600, color }}>{report.battery_condition}</div>
      <hr />
      <div>SOH: <b>{report.soh}%</b></div>
      <div>RUL: <b>{report.rul_cycles} cycles</b></div>
      <div>Predicted capacity: <b>{report.predicted_capacity} Ah</b></div>
      <div>Degradation rate: <b>{report.degradation_rate} Ah/cycle</b></div>
      <div style={{ marginTop: 8 }}>
        Est. resale value: <b>₹{report.estimated_battery_adjusted_value?.toLocaleString("en-IN")}</b>
      </div>
      <div style={{ fontSize: 11, color: "#999", marginTop: 6 }}>{report.disclaimer}</div>
    </div>
  );
}
