@echo off
rem Run the full pipeline from the repository root (package imports require it).
cd /d "%~dp0.."
python -m experts_fusion.run %*
