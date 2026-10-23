@echo off
setlocal enabledelayedexpansion

if "%EMSDK%"=="" (
    if exist "%USERPROFILE%\emsdk\emsdk_env.bat" (
        call "%USERPROFILE%\emsdk\emsdk_env.bat"
    ) else (
        echo Error: EMSDK not found. Please install and activate emsdk.
        exit /b 1
    )
)

cd /d "%~dp0\.."

if not exist web mkdir web

echo Building uneye_train.wasm and uneye_train.js...
call em++ -O3 -msimd128 -std=c++17 --bind ^
    -I cpp/app/include -I cpp/app/train/include -I cpp/include ^
    cpp/web/uneye_web.cpp cpp/app/train/src/dataset.cpp cpp/app/train/src/model_io.cpp ^
    -s MODULARIZE=1 -s EXPORT_NAME=createUnEyeModule ^
    -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=536870912 ^
    -s FILESYSTEM=0 -s ENVIRONMENT="web,worker,node" ^
    -o web/uneye_train.js

if %ERRORLEVEL% equ 0 (
    echo Build SUCCESS: web/uneye_train.js and web/uneye_train.wasm generated.
) else (
    echo Build FAILED with error code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)
