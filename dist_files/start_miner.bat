@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Pearl Miner

rem ============================================================
rem   ВПИШИТЕ СВОИ ДАННЫЕ в самой нижней строке (запуск Miner.exe) и сохраните файл:
rem     --pool    адрес пула, например prl-ru.kryptex.network:7048
rem               (другие: prl.kryptex.network:7048, prl-eu..., prl-us..., prl-sg..., prl-hk..., prl-ae...)
rem     --wallet  ваш логин на пуле: username/имя_воркера, например krxABCDEFG/rig1
rem   Заменяйте только слова ВАШ_USERNAME и worker1. Знаки процента % в строке не нужны.
rem ============================================================

if not exist "%~dp0Miner.exe" (
    echo Не найден Miner.exe рядом с батником. Распакуйте архив целиком и не переносите батник отдельно.
    pause
    exit /b 1
)

"%~dp0Miner.exe" --disable-cpu --algorithm pearlhash --stratum --pool prl-ru.kryptex.network:7048 --wallet ВАШ_USERNAME/worker1
echo.
echo Майнер остановлен (код %ERRORLEVEL%).
pause
