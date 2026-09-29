"""MongoDB connection + persistence helpers (Motor async driver)."""
import motor.motor_asyncio
import os
import datetime as _dt

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["ev_battery_rul"]

batteries_col = db["batteries"]
predictions_col = db["predictions"]
uploads_col = db["uploads"]

async def save_prediction(battery_id: str, report: dict):
    doc = {"battery_id": battery_id, "report": report, "timestamp": _dt.datetime.now(_dt.UTC)}
    await predictions_col.insert_one(doc)
    return doc

async def get_battery_history(battery_id: str):
    cursor = predictions_col.find({"battery_id": battery_id}).sort("timestamp", -1)
    return [doc async for doc in cursor]

async def save_upload_metadata(battery_id: str, filename: str, n_rows: int):
    doc = {"battery_id": battery_id, "filename": filename, "n_rows": n_rows,
           "uploaded_at": _dt.datetime.now(_dt.UTC)}
    await uploads_col.insert_one(doc)
    return doc
