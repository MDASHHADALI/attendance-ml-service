from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests

app = FastAPI(title="Attendance ML Service")

NODE_API_URL = "https://attendance-backend-qowb.onrender.com/api/attendance/ml-update"

class AttendanceData(BaseModel):
    records: list[dict]

@app.get("/")
def health_check():
    return {"status": "Attendance ML Service is running on Render"}

@app.post("/analyze")
def analyze_attendance(data: AttendanceData):
    df = pd.DataFrame(data.records)

    if len(df) < 3:
        return {"message": "Not enough data to run analysis. Need at least 3 check-ins."}

    flagged_ids = []

    # 1. Device ID Anomaly Detection (Proxy catching)
    device_counts = df.groupby('deviceId')['studentId'].nunique()
    suspicious_devices = device_counts[device_counts > 1].index

    if len(suspicious_devices) > 0:
        flagged_by_device = df[df['deviceId'].isin(suspicious_devices)]['id'].tolist()
        flagged_ids.extend(flagged_by_device)

    # 2. Geospatial Anomaly Detection (Robust Median)
    classroom_lat = df['latitude'].median()
    classroom_lon = df['longitude'].median()

    df['distance_from_center'] = np.sqrt(
        (df['latitude'] - classroom_lat)**2 + (df['longitude'] - classroom_lon)**2
    )

    threshold = 0.001
    flagged_by_location = df[df['distance_from_center'] > threshold]['id'].tolist()
    flagged_ids.extend(flagged_by_location)

    unique_flagged_ids = list(set(flagged_ids))
    class_id = df['classId'].iloc[0] if 'classId' in df.columns else None

    # 3. Send results back to Node.js backend to update Supabase
    try:
        payload = {"flaggedRecordIds": unique_flagged_ids}
        if class_id:
            payload["classId"] = class_id

        response = requests.post(NODE_API_URL, json=payload, timeout=15)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update backend: {str(e)}")

    return {
        "analyzed_count": len(df),
        "flagged_count": len(unique_flagged_ids),
        "flagged_ids": unique_flagged_ids
    }