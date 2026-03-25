@echo off
title Power 4 Trading Bot
color 0A
cls

echo.
echo  ==========================================
echo    POWER 4 TRADING BOT - Dashboard
echo  ==========================================
echo.
echo  Arrancando el dashboard...
echo  El navegador se abrira automaticamente.
echo.
echo  Para detener el bot: Ctrl+C
echo.

:: Cambiar al directorio del proyecto
cd /d "%~dp0"

:: Verificar que MetaTrader 5 esta abierto
echo  NOTA: Asegurate de que MetaTrader 5 esta abierto y logueado.
echo.

:: Lanzar Streamlit
streamlit run dashboard/app.py --server.port 8501 --server.headless false

:: Si falla, esperar antes de cerrar
if errorlevel 1 (
    echo.
    echo  ERROR: No se pudo arrancar el dashboard.
    echo  Asegurate de haber instalado las dependencias:
    echo  pip install -r requirements.txt
    echo.
    pause
)
