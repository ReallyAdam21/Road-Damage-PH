# RoadDamage PH - Philippines Road Damage Detection System

A Progressive Web App (PWA) with native mobile support for detecting and reporting road damage across the Philippines using AI-powered image analysis.

![RoadDamage PH](app/public/icon-512x512.png)

## Features

### Core Functionality
- **Interactive Map**: View road damage heatmap across the Philippines using OpenStreetMap
- **Street View Explorer**: Browse KartaView/OpenStreetCam imagery to inspect roads
- **AI Analysis**: Automatically detect potholes and cracks using YOLOv8
- **Search**: Find any location in the Philippines
- **Dashboard**: View damage statistics and recent reports

### AI Training
- **Dataset Management**: Create and manage training datasets
- **Image Annotation**: Draw bounding boxes on street images
- **Model Training**: Train custom YOLOv8 models
- **Model Management**: Switch between trained models

### Progressive Web App
- Works offline with service worker caching
- Installable on desktop and mobile devices
- Responsive design for all screen sizes
- Push notifications support

## Tech Stack

### Frontend
- React + TypeScript + Vite
- Tailwind CSS + shadcn/ui components
- Leaflet.js for maps
- PWA with Workbox

### Backend
- FastAPI (Python)
- SQLite database
- YOLOv8 for AI detection
- OpenStreetMap / Overpass API
- KartaView / OpenStreetCam API

### Native Apps
- Capacitor for Android/iOS builds

## Quick Start

### Web App (PWA)
1. Visit: https://hpwoyyac6feay.ok.kimi.link
2. Click "Install App" when prompted
3. Use offline after first load

### Development

```bash
# Install dependencies
cd app
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

### Backend

```bash
cd backend
python3 main.py
```

API will be available at http://localhost:8000

## Multi-device sync (Supabase Postgres)

By default, the backend uses a local SQLite database (data is not shared across devices).
To make the desktop + mobile apps share the same data, configure the backend to use a
hosted Postgres database (e.g. Supabase) and point the apps to your deployed backend URL.

- **Backend DB**: set `DATABASE_URL` (see `backend/.env.example`)
- **App API URL**: set `VITE_API_BASE_URL` before building (see `app/.env.example`)

### Quick steps

1. Create a Supabase project → get the Postgres connection string (SSL required).
2. Deploy the backend (Render/Fly.io/Railway/etc) and set env vars:
   - `DATABASE_URL=...`
   - `ALLOW_ORIGINS=*` (tighten later)
3. Set `VITE_API_BASE_URL=https://<your-backend-host>/api`
4. Rebuild:
   - Desktop (Tauri): build as usual
   - Android (Capacitor): `npm run build` → `npx cap sync android` → build APK

## Native App Build

### Android

```bash
cd app

# Build web assets
npm run build

# Add Android platform
npx cap add android

# Sync web assets to Android
npx cap sync android

# Open in Android Studio
npx cap open android
```

### iOS

```bash
cd app

# Build web assets
npm run build

# Add iOS platform
npx cap add ios

# Sync web assets to iOS
npx cap sync ios

# Open in Xcode
npx cap open ios
```

## AI Training Workflow

1. **Create Dataset**: Go to AI Training → Datasets → New Dataset
2. **Add Images**: Add street images from KartaView or upload your own
3. **Annotate**: Draw bounding boxes around potholes and cracks
4. **Split Dataset**: Split into train/validation/test sets
5. **Train Model**: Select dataset and train YOLOv8 model
6. **Activate Model**: Set trained model as active for detection

## Philippines-Specific Features

- Default location set to Manila, Philippines
- Search optimized for Philippine locations
- Demo data for major Philippine roads:
  - EDSA, Makati
  - C5 Road, Taguig
  - Commonwealth Avenue, QC
  - SLEX, Muntinlupa
  - And more...

## API Endpoints

### Detection
- `GET /api/roads` - Get road data from OpenStreetMap
- `GET /api/images` - Get street images from KartaView
- `POST /api/detect` - Run AI detection on image
- `GET /api/potholes` - Get pothole reports
- `GET /api/heatmap` - Get heatmap data

### Training
- `GET /api/training/datasets` - List datasets
- `POST /api/training/datasets` - Create dataset
- `POST /api/training/annotations` - Add annotations
- `POST /api/training/train` - Train model
- `GET /api/training/models` - List trained models

## Configuration

### Environment Variables
```env
# Backend
DATABASE_URL=sqlite:///database/potholes.db
MODELS_PATH=./models
DATASET_PATH=./datasets

# Frontend
VITE_API_URL=http://localhost:8000/api
```

## License

MIT License - Free for personal and commercial use.

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Acknowledgments

- OpenStreetMap contributors
- KartaView/OpenStreetCam
- Ultralytics YOLOv8
- Philippine road safety advocates
