#!/bin/bash
echo "==================================================="
echo "🚀 Starting Stock Market Forecasting GNN-LSTM App..."
echo "==================================================="
cd src
export FLASK_APP="app_user_friendly.py"
export PYTHONIOENCODING="utf-8"
python -m flask run --host=127.0.0.1 --port=5000 --no-reload
