import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export const api = axios.create({ baseURL: API_BASE });

export async function getFullReport(payload) {
  const { data } = await api.post("/predict/full-report", payload);
  const requiredFields = [
    "soh", "rul_cycles", "predicted_capacity", "degradation_rate",
    "health_score", "battery_condition", "estimated_battery_adjusted_value",
  ];
  if (!data || typeof data !== "object" || requiredFields.some((field) => !(field in data))) {
    throw new Error("The report API returned an invalid response. Rebuild the frontend with VITE_API_BASE set to the backend URL.");
  }
  return data;
}

export async function uploadBatteryCsv(file) {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post("/upload-battery-data", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getBatteryHistory(batteryId) {
  const { data } = await api.get(`/battery/${batteryId}`);
  return data;
}
