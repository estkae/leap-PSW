@echo off
REM Windows batch script to run LFM2-VL Vision Recognition Demo

echo ======================================================
echo LFM2-VL Vision Recognition POC Demo Launcher
echo ======================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Running setup...
    echo.
    python setup.py
    echo.
    echo Setup completed. Please run this script again.
    pause
    exit /b 0
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if setup is complete
if not exist "models\labels.txt" (
    echo Models not found. Running setup...
    python setup.py --skip-venv
)

echo.
echo Available demo options:
echo 1. Quick Demo (basic functionality)
echo 2. Image Processing Demo
echo 3. Batch Processing Demo
echo 4. Model Optimization Demo
echo 5. Performance Benchmark
echo 6. Interactive Menu
echo.

set /p choice="Select demo (1-6): "

if "%choice%"=="1" (
    echo Running Quick Demo...
    python quick_demo.py
) else if "%choice%"=="2" (
    echo Running Image Processing Demo...
    python examples\demo.py --demo-type image --image-path data\test_images\test_face.jpg --verbose
) else if "%choice%"=="3" (
    echo Running Batch Processing Demo...
    python examples\demo.py --demo-type batch --verbose
) else if "%choice%"=="4" (
    echo Running Model Optimization Demo...
    python examples\demo.py --demo-type optimize --target-device mobile --verbose
) else if "%choice%"=="5" (
    echo Running Performance Benchmark...
    python examples\demo.py --demo-type benchmark
) else if "%choice%"=="6" (
    echo Entering Interactive Menu...
    :menu
    cls
    echo ======================================================
    echo LFM2-VL Vision Recognition Interactive Menu
    echo ======================================================
    echo.
    echo 1. Process single image
    echo 2. Process batch of images
    echo 3. Benchmark performance
    echo 4. Test optimization
    echo 5. Run unit tests
    echo 6. View project info
    echo 0. Exit
    echo.

    set /p option="Select option (0-6): "

    if "%option%"=="1" (
        echo.
        set /p img_path="Enter image path (or press Enter for test image): "
        if "%img_path%"=="" (
            python examples\demo.py --demo-type image --image-path data\test_images\test_face.jpg --output-path output\result.jpg --verbose
        ) else (
            python examples\demo.py --demo-type image --image-path "%img_path%" --output-path output\result.jpg --verbose
        )
        echo.
        echo Result saved to: output\result.jpg
        pause
        goto menu
    ) else if "%option%"=="2" (
        python examples\demo.py --demo-type batch --verbose
        pause
        goto menu
    ) else if "%option%"=="3" (
        python examples\demo.py --demo-type benchmark
        pause
        goto menu
    ) else if "%option%"=="4" (
        echo.
        echo Available targets: mobile, ios, android, edge, tpu
        set /p target="Enter target device: "
        python examples\demo.py --demo-type optimize --target-device "%target%" --verbose
        pause
        goto menu
    ) else if "%option%"=="5" (
        echo Running unit tests...
        python -m pytest tests\ -v
        pause
        goto menu
    ) else if "%option%"=="6" (
        echo.
        echo ======================================================
        echo LFM2-VL Vision Recognition POC Information
        echo ======================================================
        echo.
        echo Version: 0.1.0 POC
        echo Platform: Windows
        echo Python:
        python --version
        echo.
        echo Project Structure:
        echo   core/            - Vision processing modules
        echo   examples/        - Demo scripts
        echo   mobile/          - iOS/Android integration
        echo   tests/           - Unit tests
        echo   models/          - Mock model files
        echo   data/            - Test images and data
        echo.
        echo For detailed documentation, see: README.md
        echo.
        pause
        goto menu
    ) else if "%option%"=="0" (
        echo Goodbye!
        goto end
    ) else (
        echo Invalid option. Please try again.
        pause
        goto menu
    )
) else (
    echo Invalid choice. Running Quick Demo...
    python quick_demo.py
)

:end
echo.
echo ======================================================
echo Demo completed! Check output/ folder for results.
echo ======================================================

REM Keep window open
pause