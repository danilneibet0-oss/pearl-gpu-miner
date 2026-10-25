@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ===== НАСТРОЙКИ =====
rem POOL   - адрес pearl-gateway на ЭТОМ компьютере (по умолчанию порт 8337).
rem          Для работы с пулом по Stratum используйте start_pool.bat.
rem WALLET - кошелёк (в режиме gateway хранится в самом gateway).
set PYTHONIOENCODING=utf-8
set POOL=127.0.0.1:8337
set WALLET=

".venv\Scripts\python.exe" miner.py --disable-cpu --algorithm pearlhash --pool %POOL% --wallet "%WALLET%"
echo.
pause
