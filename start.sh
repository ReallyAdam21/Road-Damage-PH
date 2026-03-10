#!/bin/bash

# Road Damage Detection System - Startup Script
# This script starts both the backend API and serves the frontend

echo "🚀 Starting Road Damage Detection System..."
echo ""

# Create necessary directories
mkdir -p /mnt/okcomputer/output/backend/database
mkdir -p /mnt/okcomputer/output/backend/static/images

# Start the backend API in the background
echo "📡 Starting FastAPI backend on http://localhost:8000..."
cd /mnt/okcomputer/output/backend
python3 main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

echo ""
echo "✅ Backend started successfully!"
echo ""
echo "🌐 Frontend is available at: http://localhost:8000/static/"
echo "📊 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for interrupt
trap "kill $BACKEND_PID; exit" INT
wait $BACKEND_PID
