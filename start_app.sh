#!/bin/bash

# MixMyTape Application Startup Script
# This script starts both the backend and frontend servers

echo "Starting MixMyTape Application..."

# Function to cleanup background processes on exit
cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if backend dependencies are available
if [ ! -f "cnn_gen.pt" ]; then
    echo "Error: cnn_gen.pt model file not found!"
    echo "Please ensure the instrument classification model is available."
    exit 1
fi

if [ ! -f "checkpoints/autoencoder_best.pth" ]; then
    echo "Error: autoencoder model not found!"
    echo "Please ensure the autoencoder model is available in checkpoints/autoencoder_best.pth"
    exit 1
fi

# Start backend server
echo "Starting FastAPI backend server..."
cd /Users/admin/git/MixMyTape
uvicorn audio_backend:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/docs > /dev/null; then
    echo "Backend server failed to start!"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "Backend server running on http://localhost:8000"

# Start frontend server
echo "Starting React frontend server..."
cd my-mixtape-ui
npm start &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 5

echo "Frontend server starting on http://localhost:3000"
echo ""
echo "MixMyTape is now running!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop
wait 