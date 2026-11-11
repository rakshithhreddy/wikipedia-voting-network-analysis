@echo off
title Wikipedia Voting Network Analysis - Streamlit Dashboard
echo ======================================================================
echo Wikipedia Voting Network Analysis Using Social Network Analytics
echo ======================================================================
echo Starting local Streamlit dashboard server...
echo.

cd /d "%~dp0"
python -m streamlit run app.py

pause
