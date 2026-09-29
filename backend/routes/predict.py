from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import os

from backend.models.schemas import ManualBatteryInput
from backend.services.inference import build_full_report, predict_soh_from_features
from backend.services import db

router = APIRouter()

@router.post("/predict/soh")
async def predict_soh(payload: dict):
    soh = predict_soh_from_features(payload.get("features", {}))
    return {"soh": round(soh, 2)}

@router.post("/predict/rul")
async def predict_rul(payload: ManualBatteryInput):
    report = build_full_report(payload.model_dump(exclude={"vehicle_info"}),
                                payload.vehicle_info.model_dump())
    return {"rul_cycles": report["rul_cycles"], "predicted_capacity": report["predicted_capacity"]}

@router.post("/predict/resale")
async def predict_resale(payload: ManualBatteryInput):
    report = build_full_report(payload.model_dump(exclude={"vehicle_info"}),
                                payload.vehicle_info.model_dump())
    return {
        "estimated_battery_adjusted_value": report["estimated_battery_adjusted_value"],
        "valuation_assumptions": report["valuation_assumptions"],
        "disclaimer": report["disclaimer"],
    }

@router.post("/predict/full-report")
async def predict_full_report(payload: ManualBatteryInput):
    report = build_full_report(payload.model_dump(exclude={"vehicle_info"}),
                                payload.vehicle_info.model_dump())
    if os.getenv("MONGO_URI"):
        try:
            await db.save_prediction(payload.battery_id, report)
        except Exception:
            pass
    return report

@router.post("/upload-battery-data")
async def upload_battery_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    required_min_cols = {"cycle_number"}
    if not required_min_cols.issubset(set(df.columns)):
        raise HTTPException(400, f"CSV must contain at least columns: {required_min_cols}")

    battery_id = df["battery_id"].iloc[0] if "battery_id" in df.columns else file.filename
    await db.save_upload_metadata(battery_id, file.filename, len(df))
    return {"battery_id": battery_id, "rows_ingested": len(df), "columns": list(df.columns)}

@router.get("/battery/{battery_id}")
async def get_battery(battery_id: str):
    history = await db.get_battery_history(battery_id)
    if not history:
        raise HTTPException(404, "No predictions found for this battery_id")
    return {"battery_id": battery_id, "history": history}
