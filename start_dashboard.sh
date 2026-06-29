#!/usr/bin/env bash
# O'Donoghues Dashboard Launcher
# Double-click or run from terminal to start the dashboard

cd "$(dirname "$0")" || exit 1

echo "Starting O'Donoghues Forecast Dashboard..."
# Streamlit will automatically open a browser window
python -m streamlit run dashboard/app.py
