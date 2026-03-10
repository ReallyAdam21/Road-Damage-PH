"""
AI Training Module for Road Damage Detection
Integrates YOLOv8 for training custom pothole detection models
"""

import os
import json
import sqlite3
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import random
import uuid

# Database path
DB_PATH = "/mnt/okcomputer/output/backend/database/potholes.db"
DATASET_PATH = "/mnt/okcomputer/output/backend/datasets"
MODELS_PATH = "/mnt/okcomputer/output/backend/models"

# Create directories
Path(DATASET_PATH).mkdir(parents=True, exist_ok=True)
Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)

def _get_yolo_class():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception as e:
        raise RuntimeError(
            "Model training dependencies are unavailable. Install ultralytics/torch to enable training features."
        ) from e

class DatasetManager:
    """Manages training datasets for road damage detection"""
    
    def __init__(self):
        self.dataset_path = Path(DATASET_PATH)
        self.init_database()
    
    def init_database(self):
        """Initialize dataset tables"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_images INTEGER DEFAULT 0,
                labeled_images INTEGER DEFAULT 0,
                classes TEXT,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_images (
                id TEXT PRIMARY KEY,
                dataset_id TEXT,
                image_path TEXT,
                source_url TEXT,
                latitude REAL,
                longitude REAL,
                annotation_path TEXT,
                is_labeled INTEGER DEFAULT 0,
                label_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES training_datasets(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trained_models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dataset_id TEXT,
                base_model TEXT,
                epochs INTEGER,
                batch_size INTEGER,
                map50 REAL,
                map5095 REAL,
                precision REAL,
                recall REAL,
                model_path TEXT,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 0,
                FOREIGN KEY (dataset_id) REFERENCES training_datasets(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_bank (
                id TEXT PRIMARY KEY,
                external_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'kartaview',
                image_url TEXT NOT NULL,
                latitude REAL DEFAULT 0,
                longitude REAL DEFAULT 0,
                heading REAL DEFAULT 0,
                captured_at TEXT,
                analyzed INTEGER DEFAULT 0,
                detected INTEGER DEFAULT 0,
                damage_type TEXT,
                severity_score REAL DEFAULT 0,
                confidence REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(external_id, source)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_dataset(self, name: str, description: str = "", classes: List[str] = None) -> str:
        """Create a new dataset"""
        dataset_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_datasets (id, name, description, classes)
            VALUES (?, ?, ?, ?)
        ''', (dataset_id, name, description, json.dumps(classes or ["pothole", "crack"])))
        
        conn.commit()
        conn.close()
        
        # Create dataset directories
        dataset_dir = self.dataset_path / dataset_id
        (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
        
        return dataset_id
    
    def add_image(self, dataset_id: str, image_url: str, latitude: float = 0.0, 
                  longitude: float = 0.0, image_data: bytes = None) -> str:
        """Add an image to the dataset"""
        image_id = str(uuid.uuid4())
        
        # Save image
        dataset_dir = self.dataset_path / dataset_id / "images" / "train"
        image_path = dataset_dir / f"{image_id}.jpg"
        
        if image_data:
            with open(image_path, 'wb') as f:
                f.write(image_data)
        else:
            # Download from URL
            import requests
            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading image: {e}")
                # Create a placeholder image
                try:
                    from PIL import Image
                    img = Image.new('RGB', (640, 480), color='gray')
                    img.save(image_path)
                except Exception:
                    with open(image_path, 'wb') as f:
                        f.write(b'')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO dataset_images (id, dataset_id, image_path, source_url, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_id, dataset_id, str(image_path), image_url, latitude, longitude))
        
        # Update dataset count
        cursor.execute('''
            UPDATE training_datasets 
            SET total_images = total_images + 1
            WHERE id = ?
        ''', (dataset_id,))
        
        conn.commit()
        conn.close()
        
        return image_id
    
    def add_annotation(self, image_id: str, annotations: List[Dict[str, Any]]) -> bool:
        """Add YOLO format annotations to an image"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT dataset_id, image_path FROM dataset_images WHERE id = ?', (image_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        dataset_id, image_path = row
        
        # Create label file path
        image_path = Path(image_path)
        label_path = image_path.parent.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YOLO format annotations
        with open(label_path, 'w') as f:
            for ann in annotations:
                # YOLO format: class_id x_center y_center width height (all normalized 0-1)
                f.write(f"{ann['class_id']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n")
        
        cursor.execute('''
            UPDATE dataset_images 
            SET is_labeled = 1, annotation_path = ?, label_data = ?
            WHERE id = ?
        ''', (str(label_path), json.dumps(annotations), image_id))
        
        # Update dataset labeled count
        cursor.execute('''
            UPDATE training_datasets 
            SET labeled_images = labeled_images + 1
            WHERE id = ?
        ''', (dataset_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset information"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM training_datasets WHERE id = ?', (dataset_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        dataset = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "created_at": row[3],
            "total_images": row[4],
            "labeled_images": row[5],
            "classes": json.loads(row[6]) if row[6] else ["pothole", "crack"],
            "status": row[7]
        }
        
        conn.close()
        return dataset
    
    def get_datasets(self) -> List[Dict[str, Any]]:
        """Get all datasets"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM training_datasets ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        datasets = []
        for row in rows:
            datasets.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "total_images": row[4],
                "labeled_images": row[5],
                "classes": json.loads(row[6]) if row[6] else ["pothole", "crack"],
                "status": row[7]
            })
        
        conn.close()
        return datasets
    
    def get_dataset_images(self, dataset_id: str, labeled_only: bool = False) -> List[Dict[str, Any]]:
        """Get images in a dataset"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM dataset_images WHERE dataset_id = ?'
        params = [dataset_id]
        
        if labeled_only:
            query += ' AND is_labeled = 1'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        images = []
        for row in rows:
            images.append({
                "id": row[0],
                "dataset_id": row[1],
                "image_path": row[2],
                "source_url": row[3],
                "latitude": row[4],
                "longitude": row[5],
                "annotation_path": row[6],
                "is_labeled": row[7],
                "label_data": json.loads(row[8]) if row[8] else None,
                "created_at": row[9]
            })
        
        conn.close()
        return images

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its images/annotations from disk and database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM training_datasets WHERE id = ?', (dataset_id,))
        if not cursor.fetchone():
            conn.close()
            return False

        cursor.execute(
            'SELECT image_path, annotation_path FROM dataset_images WHERE dataset_id = ?',
            (dataset_id,),
        )
        file_rows = cursor.fetchall()

        for row in file_rows:
            image_path = row[0]
            annotation_path = row[1]
            if image_path:
                try:
                    Path(image_path).unlink(missing_ok=True)
                except Exception:
                    pass
            if annotation_path:
                try:
                    Path(annotation_path).unlink(missing_ok=True)
                except Exception:
                    pass

        cursor.execute('DELETE FROM dataset_images WHERE dataset_id = ?', (dataset_id,))
        cursor.execute('DELETE FROM trained_models WHERE dataset_id = ?', (dataset_id,))
        cursor.execute('DELETE FROM training_datasets WHERE id = ?', (dataset_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        dataset_dir = self.dataset_path / dataset_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)

        return deleted

    def upsert_image_bank_entries(self, images: List[Dict[str, Any]]) -> int:
        """Insert/update fetched analyzer images into the persistent image bank."""
        if not images:
            return 0

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        count = 0

        for image in images:
            external_id = str(image.get('external_id') or image.get('id') or '')
            if not external_id:
                continue

            source = str(image.get('source') or 'kartaview')
            image_url = str(image.get('image_url') or '').strip()
            if not image_url:
                continue

            latitude = float(image.get('latitude') or image.get('lat') or 0.0)
            longitude = float(image.get('longitude') or image.get('lng') or 0.0)
            heading = float(image.get('heading') or 0.0)
            captured_at = image.get('timestamp') or image.get('captured_at')

            cursor.execute(
                '''
                INSERT INTO image_bank
                (id, external_id, source, image_url, latitude, longitude, heading, captured_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(external_id, source) DO UPDATE SET
                    image_url=excluded.image_url,
                    latitude=excluded.latitude,
                    longitude=excluded.longitude,
                    heading=excluded.heading,
                    captured_at=excluded.captured_at,
                    updated_at=CURRENT_TIMESTAMP
                ''',
                (
                    str(uuid.uuid4()),
                    external_id,
                    source,
                    image_url,
                    latitude,
                    longitude,
                    heading,
                    captured_at,
                ),
            )
            count += 1

        conn.commit()
        conn.close()
        return count

    def update_image_bank_analysis(
        self,
        external_id: str,
        source: str,
        detected: bool,
        damage_type: Optional[str] = None,
        severity_score: float = 0.0,
        confidence: float = 0.0,
    ) -> bool:
        """Update analysis metadata for an image-bank entry."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            '''
            UPDATE image_bank
            SET analyzed = 1,
                detected = ?,
                damage_type = ?,
                severity_score = ?,
                confidence = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE external_id = ? AND source = ?
            ''',
            (
                1 if detected else 0,
                damage_type,
                float(severity_score or 0.0),
                float(confidence or 0.0),
                external_id,
                source,
            ),
        )
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def get_image_bank(self, detected_only: bool = False, limit: int = 200) -> List[Dict[str, Any]]:
        """Return recently fetched/analyzed images available for dataset import."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = '''
            SELECT id, external_id, source, image_url, latitude, longitude, heading, captured_at,
                   analyzed, detected, damage_type, severity_score, confidence, created_at, updated_at
            FROM image_bank
        '''
        params: List[Any] = []
        if detected_only:
            query += ' WHERE detected = 1'
        query += ' ORDER BY updated_at DESC LIMIT ?'
        params.append(max(int(limit), 1))

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        items: List[Dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    'id': row[0],
                    'external_id': row[1],
                    'source': row[2],
                    'image_url': row[3],
                    'latitude': row[4],
                    'longitude': row[5],
                    'heading': row[6],
                    'captured_at': row[7],
                    'analyzed': bool(row[8]),
                    'detected': bool(row[9]),
                    'damage_type': row[10],
                    'severity_score': row[11],
                    'confidence': row[12],
                    'created_at': row[13],
                    'updated_at': row[14],
                }
            )
        return items

    def import_image_bank_to_dataset(self, dataset_id: str, image_bank_ids: List[str]) -> int:
        """Import selected image-bank entries into a dataset as training images."""
        if not image_bank_ids:
            return 0

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(image_bank_ids))
        cursor.execute(
            f'''
            SELECT image_url, latitude, longitude
            FROM image_bank
            WHERE id IN ({placeholders})
            ''',
            image_bank_ids,
        )
        rows = cursor.fetchall()
        conn.close()

        imported = 0
        for row in rows:
            image_url = row[0]
            latitude = float(row[1] or 0.0)
            longitude = float(row[2] or 0.0)
            if not image_url:
                continue
            self.add_image(dataset_id, image_url, latitude, longitude)
            imported += 1

        return imported
    
    def split_dataset(self, dataset_id: str, train_ratio: float = 0.7, 
                      val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Split dataset into train/val/test sets"""
        images = self.get_dataset_images(dataset_id, labeled_only=True)
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        dataset_dir = self.dataset_path / dataset_id
        
        # Move images to appropriate folders
        for img_set, folder in [(train_images, "train"), (val_images, "val"), (test_images, "test")]:
            for img in img_set:
                src_img = Path(img['image_path'])
                dst_img = dataset_dir / "images" / folder / src_img.name
                
                if src_img.exists() and src_img != dst_img:
                    shutil.move(str(src_img), str(dst_img))
                    
                    # Move corresponding label
                    if img['annotation_path']:
                        src_label = Path(img['annotation_path'])
                        dst_label = dataset_dir / "labels" / folder / src_label.name
                        if src_label.exists():
                            shutil.move(str(src_label), str(dst_label))
        
        return {
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images)
        }
    
    def create_yaml_config(self, dataset_id: str) -> str:
        """Create YOLO dataset YAML configuration"""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        
        yaml_path = self.dataset_path / dataset_id / "dataset.yaml"
        
        config = f"""path: {self.dataset_path / dataset_id}
train: images/train
val: images/val
test: images/test

nc: {len(dataset['classes'])}
names: {dataset['classes']}
"""
        
        with open(yaml_path, 'w') as f:
            f.write(config)
        
        return str(yaml_path)


class ModelTrainer:
    """Handles YOLOv8 model training"""
    
    def __init__(self):
        self.models_path = Path(MODELS_PATH)
        self.dataset_manager = DatasetManager()
    
    def train(self, dataset_id: str, model_name: str, base_model: str = "yolov8n.pt",
              epochs: int = 100, batch_size: int = 16, img_size: int = 640,
              learning_rate: float = 0.01) -> Dict[str, Any]:
        """Train a YOLOv8 model on the dataset"""
        
        # Create YAML config
        yaml_path = self.dataset_manager.create_yaml_config(dataset_id)
        if not yaml_path:
            return {"error": "Dataset not found"}
        
        # Load base model
        YOLO = _get_yolo_class()
        model = YOLO(base_model)
        
        # Train
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            project=str(self.models_path),
            name=model_name,
            exist_ok=True,
            verbose=True
        )
        
        # Save model info to database
        model_id = str(uuid.uuid4())
        model_path = self.models_path / model_name / "weights" / "best.pt"
        
        # Get metrics
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trained_models 
            (id, name, dataset_id, base_model, epochs, batch_size, 
             map50, map5095, precision, recall, model_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            model_name,
            dataset_id,
            base_model,
            epochs,
            batch_size,
            metrics.get('metrics/mAP50(B)', 0),
            metrics.get('metrics/mAP50-95(B)', 0),
            metrics.get('metrics/precision(B)', 0),
            metrics.get('metrics/recall(B)', 0),
            str(model_path)
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "model_id": model_id,
            "model_name": model_name,
            "metrics": {
                "map50": metrics.get('metrics/mAP50(B)', 0),
                "map5095": metrics.get('metrics/mAP50-95(B)', 0),
                "precision": metrics.get('metrics/precision(B)', 0),
                "recall": metrics.get('metrics/recall(B)', 0)
            }
        }
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get all trained models"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.*, d.name as dataset_name 
            FROM trained_models m
            LEFT JOIN training_datasets d ON m.dataset_id = d.id
            ORDER BY m.trained_at DESC
        ''')
        rows = cursor.fetchall()
        
        models = []
        for row in rows:
            models.append({
                "id": row[0],
                "name": row[1],
                "dataset_id": row[2],
                "dataset_name": row[12],
                "base_model": row[3],
                "epochs": row[4],
                "batch_size": row[5],
                "map50": row[6],
                "map5095": row[7],
                "precision": row[8],
                "recall": row[9],
                "model_path": row[10],
                "trained_at": row[11],
                "is_active": row[12] if len(row) > 12 else False
            })
        
        conn.close()
        return models
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Load a trained model"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT model_path FROM trained_models WHERE id = ?', (model_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row and Path(row[0]).exists():
            YOLO = _get_yolo_class()
            return YOLO(row[0])
        return None
    
    def set_active_model(self, model_id: str):
        """Set a model as the active model for inference"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Deactivate all models
        cursor.execute('UPDATE trained_models SET is_active = 0')
        
        # Activate selected model
        cursor.execute('UPDATE trained_models SET is_active = 1 WHERE id = ?', (model_id,))
        
        conn.commit()
        conn.close()
    
    def get_active_model(self) -> Optional[str]:
        """Get the ID of the active model"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM trained_models WHERE is_active = 1')
        row = cursor.fetchone()
        
        conn.close()
        
        return row[0] if row else None
    
    def evaluate(self, model_id: str, dataset_id: str = None) -> Dict[str, Any]:
        """Evaluate a model on a dataset"""
        model = self.get_model(model_id)
        if not model:
            return {"error": "Model not found"}
        
        if dataset_id:
            yaml_path = self.dataset_manager.create_yaml_config(dataset_id)
            if yaml_path:
                results = model.val(data=yaml_path)
                return {
                    "map50": results.results_dict.get('metrics/mAP50(B)', 0),
                    "map5095": results.results_dict.get('metrics/mAP50-95(B)', 0),
                    "precision": results.results_dict.get('metrics/precision(B)', 0),
                    "recall": results.results_dict.get('metrics/recall(B)', 0)
                }
        
        return {"error": "Evaluation failed"}
    
    def predict(self, model_id: str, image_path: str, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Run inference on an image"""
        model = self.get_model(model_id)
        if not model:
            return []
        
        results = model(image_path, conf=conf_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detections.append({
                    "class_id": int(box.cls),
                    "class_name": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3]),
                    "x_center": float(box.xywh[0][0]),
                    "y_center": float(box.xywh[0][1]),
                    "width": float(box.xywh[0][2]),
                    "height": float(box.xywh[0][3])
                })
        
        return detections


# Global instances
dataset_manager = DatasetManager()
model_trainer = ModelTrainer()
