"""
Road Damage Detection System - FastAPI Backend
Integrates OpenStreetMap, KartaView, AI Detection, and Heatmap Visualization
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import os
import requests
import random
import uuid
import math
from pathlib import Path

from db import create_db_engine, init_db, fetch_all, fetch_one, execute, executemany

# Load local env vars from backend/.env (deployment platforms use real env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Create directories (local/dev friendly)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = Path(os.getenv("ROADDAMAGE_STATIC_DIR") or (BASE_DIR / "static"))
STATIC_IMAGES_DIR = STATIC_DIR / "images"
STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Road Damage Detection API", version="1.0.0", docs_url="/docs", openapi_url="/openapi.json")

# Serve frontend static files
frontend_path = "/mnt/okcomputer/output/app/dist"
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Warning: Frontend build not found at {frontend_path}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    # For production, set ALLOW_ORIGINS="https://your-frontend-domain,capacitor://localhost"
    allow_origins=[o.strip() for o in (os.getenv("ALLOW_ORIGINS") or "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database (SQLite by default, Supabase Postgres via DATABASE_URL)
engine = create_db_engine()

# ==================== DATABASE SETUP ====================

init_db(engine)

# ==================== PHILIPPINES MOCK DATA ====================

def seed_philippines_data():
    """Add mock data for Philippines roads"""
    # Check if data already exists
    existing = fetch_one(engine, "SELECT COUNT(*)::int as count FROM pothole_reports") or {"count": 0}
    if int(existing.get("count") or 0) > 0:
        return
    
    # Philippines road data
    philippines_roads = [
        (14.5995, 120.9842, "EDSA, Makati City"),
        (14.5547, 121.0244, "C5 Road, Taguig City"),
        (14.6760, 121.0437, "Commonwealth Avenue, Quezon City"),
        (14.5176, 121.0509, "SLEX, Muntinlupa City"),
        (14.6091, 121.0223, "Ortigas Avenue, Pasig City"),
        (14.4807, 121.0410, "Alabang-Zapote Road, Las Piñas"),
        (14.5794, 121.0359, "Marcos Highway, Marikina"),
        (14.4123, 120.9423, "Aguinaldo Highway, Cavite"),
        (14.5458, 121.0685, "J.P. Rizal Street, Makati"),
        (14.6323, 121.0013, "Rizal Avenue, Manila"),
        (14.4502, 120.9820, "Taal Vista, Tagaytay"),
        (14.3430, 121.0647, "Sta. Rosa-Tagaytay Road"),
        (14.8167, 120.2833, "Olongapo-Gapan Road"),
        (15.1401, 120.5878, "MacArthur Highway, Pampanga"),
        (14.0583, 121.3244, "Calamba Pagsanjan Road"),
    ]
    
    payload = []
    for lat, lng, road_name in philippines_roads:
        report_id = str(uuid.uuid4())
        severity = random.uniform(0.3, 0.95)
        damage_type = "pothole" if severity > 0.5 else "crack"
        payload.append(
            {
                "id": report_id,
                "latitude": lat + random.uniform(-0.005, 0.005),
                "longitude": lng + random.uniform(-0.005, 0.005),
                "road_name": road_name,
                "damage_type": damage_type,
                "severity_score": round(severity, 2),
                "confidence": round(random.uniform(0.7, 0.95), 2),
                "image_path": None,
                "ocr_metadata": None,
                "status": "active",
            }
        )

    executemany(
        engine,
        """
        INSERT INTO pothole_reports
          (id, latitude, longitude, road_name, damage_type, severity_score, confidence, image_path, ocr_metadata, status)
        VALUES
          (:id, :latitude, :longitude, :road_name, :damage_type, :severity_score, :confidence, :image_path, :ocr_metadata, :status)
        """,
        payload,
    )
    print("Philippines demo data seeded successfully!")

# Seed data on startup
seed_philippines_data()

# ==================== PYDANTIC MODELS ====================

class PotholeReport(BaseModel):
    id: str
    latitude: float
    longitude: float
    road_name: Optional[str]
    damage_type: str
    severity_score: float
    confidence: float
    image_path: Optional[str]
    detected_at: str
    ocr_metadata: Optional[Dict[str, Any]]
    status: str

class PotholeCreate(BaseModel):
    latitude: float
    longitude: float
    road_name: Optional[str] = None
    damage_type: str = "pothole"
    severity_score: float
    confidence: float
    image_path: Optional[str] = None
    ocr_metadata: Optional[Dict[str, Any]] = None

class RoadSegment(BaseModel):
    id: str
    name: Optional[str]
    coordinates: List[List[float]]
    length: Optional[float]
    highway_type: Optional[str]

class DetectionResult(BaseModel):
    detected: bool
    damage_type: Optional[str]
    severity_score: float
    confidence: float
    bounding_boxes: Optional[List[Dict[str, Any]]]

class KartaViewImage(BaseModel):
    id: str
    lat: float
    lng: float
    sequence_id: str
    timestamp: Optional[str]
    image_url: str

# ==================== OPENS STREET MAP API ====================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def fetch_road_data_osm(bbox: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Fetch road data from OpenStreetMap using Overpass API
    bbox: {south, west, north, east}
    """
    query = f'''
    [out:json][timeout:25];
    (
        way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]
        ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out body;
    >;
    out skel qt;
    '''
    
    try:
        response = requests.get(OVERPASS_URL, params={'data': query}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        roads = []
        nodes = {}
        
        # First pass: collect all nodes
        for element in data.get('elements', []):
            if element['type'] == 'node':
                nodes[element['id']] = [element['lat'], element['lon']]
        
        # Second pass: process ways (roads)
        for element in data.get('elements', []):
            if element['type'] == 'way' and 'highway' in element.get('tags', {}):
                road_coords = []
                for node_id in element.get('nodes', []):
                    if node_id in nodes:
                        road_coords.append(nodes[node_id])
                
                if road_coords:
                    roads.append({
                        'id': str(element['id']),
                        'name': element.get('tags', {}).get('name', 'Unnamed Road'),
                        'highway_type': element['tags']['highway'],
                        'coordinates': road_coords,
                        'one_way': element.get('tags', {}).get('oneway', 'no') == 'yes'
                    })
        
        return roads
    except Exception as e:
        print(f"Error fetching OSM data: {e}")
        return []

# ==================== KARTAVIEW API ====================

KARTAVIEW_API_URL = "https://api.openstreetcam.org/2.0"
MAPILLARY_API_URL = "https://graph.mapillary.com/images"

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _resolve_kartaview_image_url(photo: Dict[str, Any]) -> str:
    """
    Pick the best available image URL from KartaView payload fields.
    """
    candidates = [
        photo.get("imageProcUrl"),
        photo.get("fileurlProc"),
        photo.get("imageLthUrl"),
        photo.get("imageThUrl"),
        photo.get("fileurlLTh"),
        photo.get("fileurlTh"),
        photo.get("fileurl"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        url = str(candidate)
        if "{{sizeprefix}}" in url:
            url = url.replace("{{sizeprefix}}", "wrapped_proc")
        return url
    return ""

def _get_mapillary_token() -> str:
    # Support both names to keep env setup flexible.
    return os.getenv("MAPILLARY_TOKEN") or os.getenv("MAPILLARY_ACCESS_TOKEN") or ""

def _radius_to_bbox(lat: float, lng: float, radius_m: int) -> str:
    """
    Convert radius in meters to a lon/lat bbox string for Mapillary Graph API.
    """
    radius = max(int(radius_m), 50)
    lat_delta = radius / 111320.0
    cos_lat = max(abs(math.cos(math.radians(lat))), 0.1)
    lng_delta = radius / (111320.0 * cos_lat)
    return f"{lng - lng_delta},{lat - lat_delta},{lng + lng_delta},{lat + lat_delta}"

def fetch_mapillary_images(lat: float, lng: float, radius: int = 500, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch fallback street-level images from Mapillary Graph API.
    """
    images: List[Dict[str, Any]] = []
    mapillary_token = _get_mapillary_token()
    if not mapillary_token:
        print("MAPILLARY_TOKEN not set; skipping Mapillary fallback for street images")
        return images

    try:
        params = {
            # geometry.coordinates are [lon, lat]
            "fields": "id,geometry,thumb_1024_url,captured_at,compass_angle",
            "bbox": _radius_to_bbox(lat, lng, radius),
            "limit": limit,
            "access_token": mapillary_token,
        }
        response = requests.get(MAPILLARY_API_URL, params=params, timeout=10)
        if response.status_code != 200:
            print(f"Mapillary API HTTP {response.status_code}: {response.text[:200]}...")
            return images

        data = response.json()
        for item in data.get("data", []):
            coords = ((item.get("geometry") or {}).get("coordinates") or [lng, lat])
            if len(coords) < 2:
                continue
            image_url = item.get("thumb_1024_url")
            if not image_url:
                continue
            lon, la = coords[0], coords[1]
            images.append(
                {
                    "id": str(item.get("id", uuid.uuid4())),
                    "lat": _safe_float(la, lat),
                    "lng": _safe_float(lon, lng),
                    "sequence_id": "",
                    "timestamp": item.get("captured_at"),
                    "image_url": image_url,
                    "heading": int(_safe_float(item.get("compass_angle"), 0.0)),
                    "source": "mapillary",
                }
            )
    except Exception as e:
        print(f"Mapillary API error: {e}")

    return images

def fetch_kartaview_images(lat: float, lng: float, radius: int = 500, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch street-level images from KartaView/OpenStreetCam API
    """
    # Try multiple APIs for better coverage
    images = []
    
    # Try KartaView API first
    try:
        url = f"{KARTAVIEW_API_URL}/photo/"
        params = {
            'lat': lat,
            'lng': lng,
            'radius': radius,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'result' in data and 'data' in data['result']:
                for photo in data['result']['data']:
                    image_url = _resolve_kartaview_image_url(photo)
                    if not image_url:
                        continue
                    images.append({
                        'id': str(photo.get('id', uuid.uuid4())),
                        'lat': _safe_float(photo.get('lat'), lat),
                        'lng': _safe_float(photo.get('lng'), lng),
                        'sequence_id': str(photo.get('sequenceId') or photo.get('sequence_id') or ''),
                        'timestamp': photo.get('shotDate') or photo.get('dateAdded'),
                        'image_url': image_url,
                        'heading': int(_safe_float(photo.get('heading'), 0.0)),
                        'source': 'kartaview'
                    })
    except Exception as e:
        print(f"KartaView API error: {e}")
    
    # If no images found from KartaView, use Mapillary as fallback.
    if len(images) == 0:
        images = fetch_mapillary_images(lat, lng, radius, limit)
        if len(images) > 0:
            print(f"Using Mapillary fallback: {len(images)} images")
    
    # Do not generate fake images; let the frontend show an empty-state instead.
    if len(images) == 0:
        print("No real street images found for this location")
    
    return images

# ==================== AI DETECTION SERVICE ====================

def detect_pothole_ai(image_path: str) -> DetectionResult:
    """
    Simulated AI pothole detection using YOLOv8
    In production, this would use actual YOLOv8 model
    """
    # Simulate detection with random results for demonstration
    detection_confidence = random.random()
    
    if detection_confidence > 0.3:  # 70% chance of detection
        severity = random.uniform(0.3, 1.0)
        return DetectionResult(
            detected=True,
            damage_type="pothole" if severity > 0.5 else "crack",
            severity_score=round(severity, 2),
            confidence=round(detection_confidence, 2),
            bounding_boxes=[{
                "x": random.randint(100, 400),
                "y": random.randint(100, 300),
                "width": random.randint(50, 150),
                "height": random.randint(30, 100),
                "confidence": round(detection_confidence, 2)
            }]
        )
    else:
        return DetectionResult(
            detected=False,
            damage_type=None,
            severity_score=0.0,
            confidence=round(detection_confidence, 2),
            bounding_boxes=[]
        )

# ==================== OCR SERVICE ====================

def extract_ocr_metadata(image_path: str) -> Dict[str, Any]:
    """
    Simulated OCR metadata extraction
    In production, this would use Tesseract OCR
    """
    # Simulate OCR extraction
    road_signs = ["Main St", "Highway 101", "Oak Avenue", "Park Road", "Elm Street"]
    
    return {
        'road_name': random.choice(road_signs) if random.random() > 0.5 else None,
        'speed_limit': random.choice([25, 35, 45, 55, 65]) if random.random() > 0.7 else None,
        'detected_signs': random.randint(0, 3),
        'timestamp_confidence': round(random.uniform(0.7, 1.0), 2)
    }

def reverse_geocode_road_name(lat: float, lng: float) -> Optional[str]:
    """
    Resolve a human-friendly road/location label from coordinates.
    """
    if abs(lat) < 0.000001 and abs(lng) < 0.000001:
        return None

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "format": "jsonv2",
                "lat": lat,
                "lon": lng,
                "zoom": 18,
                "addressdetails": 1,
            },
            headers={"User-Agent": "RoadDamagePH/1.0"},
            timeout=8,
        )
        if response.status_code != 200:
            return None

        data = response.json()
        address = data.get("address", {}) if isinstance(data, dict) else {}
        road = address.get("road") or address.get("pedestrian") or address.get("footway")
        locality = (
            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("village")
            or address.get("suburb")
        )

        if road and locality:
            return f"{road}, {locality}"
        if road:
            return str(road)
        display_name = data.get("display_name") if isinstance(data, dict) else None
        if display_name:
            return str(display_name).split(",")[0].strip()
    except Exception as e:
        print(f"Reverse geocode error: {e}")

    return None

# ==================== API ENDPOINTS ====================

@app.get("/")
def read_root():
    """Redirect to frontend"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/")

@app.get("/api")
def api_info():
    """API Information"""
    return {
        "message": "Road Damage Detection API",
        "version": "1.0.0",
        "endpoints": {
            "roads": "/api/roads",
            "images": "/api/images",
            "detect": "/api/detect",
            "potholes": "/api/potholes",
            "heatmap": "/api/heatmap",
            "stats": "/api/stats"
        }
    }

@app.get("/api/roads")
def get_roads(
    south: float = -34.6037,
    west: float = -58.3816,
    north: float = -34.5837,
    east: float = -58.3616
):
    """Get road data from OpenStreetMap for a bounding box"""
    bbox = {"south": south, "west": west, "north": north, "east": east}
    roads = fetch_road_data_osm(bbox)
    return {"count": len(roads), "roads": roads}

@app.get("/api/images")
def get_images(
    lat: float = -34.6037,
    lng: float = -58.3816,
    radius: int = 500,
    limit: int = 50
):
    """Get street-level images from KartaView"""
    images = fetch_kartaview_images(lat, lng, radius, limit)
    return {"count": len(images), "images": images}

@app.get("/api/image-proxy")
def proxy_image(image_url: str):
    """Proxy remote image URLs so frontend rendering is more reliable across web/mobile."""
    normalized_url = image_url.strip()
    if normalized_url.startswith("//"):
        normalized_url = f"https:{normalized_url}"
    elif normalized_url.startswith("http://"):
        normalized_url = normalized_url.replace("http://", "https://", 1)

    try:
        upstream = requests.get(
            normalized_url,
            timeout=20,
            headers={"User-Agent": "RoadDamagePH/1.0"},
        )
        upstream.raise_for_status()
        media_type = upstream.headers.get("content-type", "image/jpeg")
        return Response(
            content=upstream.content,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {e}")

@app.post("/api/detect")
def detect_damage(
    image_url: str,
    lat: float = 0.0,
    lng: float = 0.0,
    background_tasks: BackgroundTasks = None,
):
    """Run AI detection on an image and store a pothole report if damage is found.

    The optional lat/lng parameters allow the frontend to associate detections
    with the correct map location so they appear in the damage statistics and heatmap.
    """
    # In production, download and process actual image
    result = detect_pothole_ai(image_url)

    if result.detected:
        # Extract OCR metadata
        ocr_data = extract_ocr_metadata(image_url)
        resolved_road_name = reverse_geocode_road_name(lat, lng)
        road_name = resolved_road_name or ocr_data.get("road_name")

        # Store in database
        report_id = str(uuid.uuid4())
        execute(
            engine,
            """
            INSERT INTO pothole_reports
              (id, latitude, longitude, road_name, damage_type, severity_score, confidence, image_path, ocr_metadata, status)
            VALUES
              (:id, :latitude, :longitude, :road_name, :damage_type, :severity_score, :confidence, :image_path, :ocr_metadata, :status)
            """,
            {
                "id": report_id,
                "latitude": lat,
                "longitude": lng,
                "road_name": road_name,
                "damage_type": result.damage_type,
                "severity_score": result.severity_score,
                "confidence": result.confidence,
                "image_path": image_url,
                "ocr_metadata": ocr_data,
                "status": "active",
            },
        )

        return {
            "detection": result.dict(),
            "ocr_metadata": ocr_data,
            "report_id": report_id,
            "stored": True,
        }

    return {"detection": result.dict(), "stored": False}

@app.get("/api/potholes")
def get_potholes(
    south: Optional[float] = None,
    west: Optional[float] = None,
    north: Optional[float] = None,
    east: Optional[float] = None,
    min_severity: float = 0.0,
    limit: int = 1000
):
    """Get pothole reports with optional bounding box filter"""
    sql = """
        SELECT id, latitude, longitude, road_name, damage_type,
               severity_score, confidence, image_path, detected_at,
               ocr_metadata, status
        FROM pothole_reports
        WHERE severity_score >= :min_severity
    """
    params: Dict[str, Any] = {"min_severity": min_severity, "limit": limit}
    if all(v is not None for v in [south, west, north, east]):
        sql += " AND latitude BETWEEN :south AND :north AND longitude BETWEEN :west AND :east"
        params.update({"south": south, "north": north, "west": west, "east": east})
    sql += " ORDER BY detected_at DESC NULLS LAST LIMIT :limit"

    potholes = fetch_all(engine, sql, params)
    return {"count": len(potholes), "potholes": potholes}

@app.get("/api/heatmap")
def get_heatmap_data(
    south: Optional[float] = None,
    west: Optional[float] = None,
    north: Optional[float] = None,
    east: Optional[float] = None
):
    """Get heatmap data for visualization"""
    sql = """
        SELECT latitude as lat, longitude as lng, severity_score as intensity, damage_type as type
        FROM pothole_reports
        WHERE status = 'active'
    """
    params: Dict[str, Any] = {}
    if all(v is not None for v in [south, west, north, east]):
        sql += " AND latitude BETWEEN :south AND :north AND longitude BETWEEN :west AND :east"
        params.update({"south": south, "north": north, "west": west, "east": east})

    heatmap_points = fetch_all(engine, sql, params)
    
    return {
        "count": len(heatmap_points),
        "max_intensity": 1.0,
        "points": heatmap_points
    }

@app.post("/api/potholes")
def create_pothole_report(report: PotholeCreate):
    """Manually create a pothole report"""
    report_id = str(uuid.uuid4())

    execute(
        engine,
        """
        INSERT INTO pothole_reports
          (id, latitude, longitude, road_name, damage_type, severity_score, confidence, image_path, ocr_metadata, status)
        VALUES
          (:id, :latitude, :longitude, :road_name, :damage_type, :severity_score, :confidence, :image_path, :ocr_metadata, :status)
        """,
        {
            "id": report_id,
            "latitude": report.latitude,
            "longitude": report.longitude,
            "road_name": report.road_name,
            "damage_type": report.damage_type,
            "severity_score": report.severity_score,
            "confidence": report.confidence,
            "image_path": report.image_path,
            "ocr_metadata": report.ocr_metadata,
            "status": "active",
        },
    )
    
    return {"id": report_id, "message": "Report created successfully"}

@app.delete("/api/potholes/{report_id}")
def delete_pothole_report(report_id: str):
    """Delete a pothole report"""
    existing = fetch_one(engine, "SELECT id FROM pothole_reports WHERE id = :id", {"id": report_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Report not found")
    execute(engine, "DELETE FROM pothole_reports WHERE id = :id", {"id": report_id})
    return {"message": "Report deleted successfully"}

@app.get("/api/stats")
def get_statistics():
    """Get system statistics"""
    total = fetch_one(engine, "SELECT COUNT(*)::int as total_reports FROM pothole_reports") or {"total_reports": 0}
    total_reports = int(total.get("total_reports") or 0)

    sev_rows = fetch_all(
        engine,
        """
        SELECT
          CASE
            WHEN severity_score >= 0.8 THEN 'critical'
            WHEN severity_score >= 0.5 THEN 'moderate'
            ELSE 'minor'
          END as severity_level,
          COUNT(*)::int as count
        FROM pothole_reports
        GROUP BY severity_level
        """,
    )
    severity_counts = {r["severity_level"]: r["count"] for r in sev_rows}

    type_rows = fetch_all(
        engine,
        "SELECT damage_type, COUNT(*)::int as count FROM pothole_reports GROUP BY damage_type",
    )
    type_counts = {r["damage_type"]: r["count"] for r in type_rows}
    
    return {
        "total_reports": total_reports,
        "severity_distribution": severity_counts,
        "damage_type_distribution": type_counts
    }

# ==================== AI TRAINING API (OPTIONAL) ====================
#
# The training module depends on heavy optional deps (ultralytics/torch). We want the
# core API (maps/images/detect/stats) to run even if those aren't installed.
try:
    from ai_training import dataset_manager, model_trainer
    TRAINING_AVAILABLE = True
except Exception as e:
    print(f"Warning: training module unavailable (ai_training): {e}")
    dataset_manager = None
    model_trainer = None
    TRAINING_AVAILABLE = False

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    classes: Optional[List[str]] = None

class AnnotationCreate(BaseModel):
    image_id: str
    annotations: List[Dict[str, Any]]

class TrainingConfig(BaseModel):
    dataset_id: str
    model_name: str
    base_model: Optional[str] = "yolov8n.pt"
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 16
    img_size: Optional[int] = 640
    learning_rate: Optional[float] = 0.01

class ImageBankEntryIn(BaseModel):
    external_id: str
    source: Optional[str] = "kartaview"
    image_url: str
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    heading: Optional[float] = 0.0
    timestamp: Optional[str] = None

class ImageBankBatchIn(BaseModel):
    images: List[ImageBankEntryIn]

class ImageBankAnalysisIn(BaseModel):
    external_id: str
    source: Optional[str] = "kartaview"
    detected: bool
    damage_type: Optional[str] = None
    severity_score: Optional[float] = 0.0
    confidence: Optional[float] = 0.0

class ImageBankImportIn(BaseModel):
    image_bank_ids: List[str]


def _ensure_training_available():
    if not TRAINING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI training endpoints are disabled because optional dependencies "
                "are not installed. Install ultralytics (and torch if needed) to enable."
            ),
        )

@app.get("/api/training/datasets")
def get_datasets():
    """Get all training datasets"""
    _ensure_training_available()
    return {"datasets": dataset_manager.get_datasets()}

@app.post("/api/training/datasets")
def create_dataset(dataset: DatasetCreate):
    """Create a new training dataset"""
    _ensure_training_available()
    dataset_id = dataset_manager.create_dataset(
        name=dataset.name,
        description=dataset.description,
        classes=dataset.classes
    )
    return {"id": dataset_id, "message": "Dataset created successfully"}

@app.get("/api/training/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    """Get dataset details"""
    _ensure_training_available()
    dataset = dataset_manager.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    images = dataset_manager.get_dataset_images(dataset_id)
    return {**dataset, "images": images}

@app.delete("/api/training/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Delete a dataset and all associated images/labels."""
    _ensure_training_available()
    deleted = dataset_manager.delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": "Dataset deleted successfully"}

@app.post("/api/training/datasets/{dataset_id}/images")
def add_image_to_dataset(dataset_id: str, image_url: str, latitude: float = 0.0, longitude: float = 0.0):
    """Add an image to the dataset"""
    _ensure_training_available()
    image_id = dataset_manager.add_image(dataset_id, image_url, latitude, longitude)
    return {"id": image_id, "message": "Image added successfully"}

@app.post("/api/training/datasets/{dataset_id}/images/upload")
async def upload_images_to_dataset(
    dataset_id: str,
    files: List[UploadFile] = File(...),
    latitude: float = 0.0,
    longitude: float = 0.0,
):
    """Upload one or more local image files directly to a training dataset."""
    _ensure_training_available()

    if not files:
        raise HTTPException(status_code=400, detail="No image files provided")

    uploaded_ids: List[str] = []
    failed: List[Dict[str, str]] = []

    for upload in files:
        filename = upload.filename or "uploaded-image.jpg"
        try:
            content = await upload.read()
            if not content:
                failed.append({"filename": filename, "error": "Empty file"})
                continue

            image_id = dataset_manager.add_image(
                dataset_id=dataset_id,
                image_url=filename,
                latitude=latitude,
                longitude=longitude,
                image_data=content,
            )
            uploaded_ids.append(image_id)
        except Exception as e:
            failed.append({"filename": filename, "error": str(e)})

    if len(uploaded_ids) == 0:
        raise HTTPException(status_code=400, detail="Failed to upload provided images")

    return {
        "message": f"Uploaded {len(uploaded_ids)} image(s)",
        "count": len(uploaded_ids),
        "ids": uploaded_ids,
        "failed": failed,
    }

@app.get("/api/training/image-bank")
def get_image_bank(detected_only: bool = False, limit: int = 200):
    """List fetched/analyzed images available for dataset import."""
    _ensure_training_available()
    images = dataset_manager.get_image_bank(detected_only=detected_only, limit=limit)
    return {"count": len(images), "images": images}

@app.post("/api/training/image-bank/batch")
def save_image_bank_batch(payload: ImageBankBatchIn):
    """Save or update fetched image metadata in image bank."""
    _ensure_training_available()
    count = dataset_manager.upsert_image_bank_entries([item.dict() for item in payload.images])
    return {"message": f"Saved {count} image(s)", "count": count}

@app.post("/api/training/image-bank/analysis")
def update_image_bank_analysis(payload: ImageBankAnalysisIn):
    """Store analysis results for an image-bank item."""
    _ensure_training_available()
    updated = dataset_manager.update_image_bank_analysis(
        external_id=payload.external_id,
        source=payload.source or "kartaview",
        detected=payload.detected,
        damage_type=payload.damage_type,
        severity_score=payload.severity_score or 0.0,
        confidence=payload.confidence or 0.0,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Image bank entry not found")
    return {"message": "Image analysis updated"}

@app.post("/api/training/datasets/{dataset_id}/import-image-bank")
def import_image_bank_to_dataset(dataset_id: str, payload: ImageBankImportIn):
    """Import selected image-bank entries into a dataset."""
    _ensure_training_available()
    imported_count = dataset_manager.import_image_bank_to_dataset(dataset_id, payload.image_bank_ids)
    return {"message": f"Imported {imported_count} image(s)", "count": imported_count}

@app.post("/api/training/annotations")
def add_annotation(annotation: AnnotationCreate):
    """Add annotations to an image"""
    _ensure_training_available()
    success = dataset_manager.add_annotation(annotation.image_id, annotation.annotations)
    if not success:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"message": "Annotation added successfully"}

@app.post("/api/training/datasets/{dataset_id}/split")
def split_dataset(dataset_id: str, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
    """Split dataset into train/val/test sets"""
    _ensure_training_available()
    result = dataset_manager.split_dataset(dataset_id, train_ratio, val_ratio, test_ratio)
    return {"message": "Dataset split successfully", "split": result}

@app.post("/api/training/train")
def train_model(config: TrainingConfig, background_tasks: BackgroundTasks = None):
    """Start model training"""
    _ensure_training_available()
    try:
        result = model_trainer.train(
            dataset_id=config.dataset_id,
            model_name=config.model_name,
            base_model=config.base_model,
            epochs=config.epochs,
            batch_size=config.batch_size,
            img_size=config.img_size,
            learning_rate=config.learning_rate
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/models")
def get_models():
    """Get all trained models"""
    _ensure_training_available()
    return {"models": model_trainer.get_models()}

@app.post("/api/training/models/{model_id}/activate")
def activate_model(model_id: str):
    """Set a model as active"""
    _ensure_training_available()
    model_trainer.set_active_model(model_id)
    return {"message": "Model activated successfully"}

@app.get("/api/training/models/active")
def get_active_model():
    """Get the active model ID"""
    _ensure_training_available()
    model_id = model_trainer.get_active_model()
    return {"model_id": model_id}

@app.post("/api/training/models/{model_id}/evaluate")
def evaluate_model(model_id: str, dataset_id: Optional[str] = None):
    """Evaluate a model"""
    _ensure_training_available()
    result = model_trainer.evaluate(model_id, dataset_id)
    return result

@app.post("/api/training/predict")
def predict_with_model(model_id: str, image_path: str, conf_threshold: float = 0.25):
    """Run inference with a specific model"""
    _ensure_training_available()
    detections = model_trainer.predict(model_id, image_path, conf_threshold)
    return {"detections": detections}

# Mount backend static files (images)
app.mount("/api/static", StaticFiles(directory=str(STATIC_DIR)), name="backend_static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
