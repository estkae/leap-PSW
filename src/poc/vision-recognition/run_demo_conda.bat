@echo off
REM Windows batch script to run LFM2-VL Vision Recognition Demo
REM Using Conda environment: leap-PSW

echo ======================================================
echo LFM2-VL Vision Recognition POC Demo Launcher
echo Using Conda Environment: leap-PSW
echo ======================================================
echo.

REM Check if Conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda from https://conda.io
    pause
    exit /b 1
)

REM Check if leap-PSW environment exists
conda env list | findstr /C:"leap-PSW" >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda environment 'leap-PSW' not found.
    echo.
    echo Creating new Conda environment...
    call conda create -n leap-PSW python=3.10 -y
    echo.
    echo Environment created. Installing dependencies...
    call conda activate leap-PSW

    REM Install basic requirements
    echo Installing core packages...
    call conda install numpy opencv pillow scikit-image -y
    call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

    REM Install additional packages via pip
    if exist "config\requirements.txt" (
        echo Installing additional requirements...
        pip install -r config\requirements.txt
    )

    echo.
    echo Setup completed!
) else (
    echo Found existing leap-PSW environment.
)

REM Activate Conda environment
echo.
echo Activating Conda environment: leap-PSW...
call conda activate leap-PSW

REM Verify activation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate Conda environment
    pause
    exit /b 1
)

echo Environment activated successfully!
python --version
echo.

REM Check if setup is complete
if not exist "models\labels.txt" (
    echo Models not found. Running initial setup...
    python quick_setup.py
    echo.
)

REM Main Menu
echo Available demo options:
echo 1. Quick Demo (basic functionality test)
echo 2. Image Processing Demo
echo 3. Batch Processing Demo
echo 4. Model Optimization Demo
echo 5. Performance Benchmark
echo 6. Interactive Menu
echo 7. Install/Update Dependencies
echo.

set /p choice="Select demo (1-7): "

if "%choice%"=="1" (
    echo.
    echo Running Quick Demo...
    echo ======================================================
    python simple_test.py
) else if "%choice%"=="2" (
    echo.
    echo Running Image Processing Demo...
    echo ======================================================
    python examples\demo.py --demo-type image --image-path data\test_images\test_face.jpg --verbose
) else if "%choice%"=="3" (
    echo.
    echo Running Batch Processing Demo...
    echo ======================================================
    python examples\demo.py --demo-type batch --verbose
) else if "%choice%"=="4" (
    echo.
    echo Running Model Optimization Demo...
    echo ======================================================
    python examples\demo.py --demo-type optimize --target-device mobile --verbose
) else if "%choice%"=="5" (
    echo.
    echo Running Performance Benchmark...
    echo ======================================================
    python examples\demo.py --demo-type benchmark
) else if "%choice%"=="6" (
    echo Entering Interactive Menu...
    :menu
    cls
    echo ======================================================
    echo LFM2-VL Vision Recognition Interactive Menu
    echo Conda Environment: leap-PSW
    echo ======================================================
    echo.
    echo 1. Process single image
    echo 2. Process batch of images
    echo 3. Benchmark performance
    echo 4. Test optimization
    echo 5. Run unit tests
    echo 6. View project info
    echo 7. Check environment packages
    echo 8. Update dependencies
    echo 0. Exit
    echo.

    set /p option="Select option (0-8): "

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
        echo.
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
        echo Environment: Conda leap-PSW
        echo Python:
        python --version
        echo.
        echo Conda Info:
        conda info --envs | findstr leap-PSW
        echo.
        echo Project Structure:
        echo   core/            - Vision processing modules
        echo   examples/        - Demo scripts
        echo   mobile/          - iOS/Android integration
        echo   tests/           - Unit tests
        echo   models/          - Mock model files
        echo   data/            - Test images and data
        echo   config/          - Configuration and requirements
        echo.
        echo Key Features:
        echo   - LFM2-VL Vision Pipeline
        echo   - Face and Object Detection
        echo   - Model Optimization for Edge
        echo   - Mobile Integration (iOS/Android)
        echo   - Performance Benchmarking
        echo.
        echo For detailed documentation, see: README.md
        echo.
        pause
        goto menu
    ) else if "%option%"=="7" (
        echo.
        echo Installed packages in leap-PSW environment:
        echo ======================================================
        conda list
        echo.
        pause
        goto menu
    ) else if "%option%"=="8" (
        echo.
        echo Updating dependencies...
        echo ======================================================
        pip install --upgrade -r config\requirements.txt
        echo.
        echo Update completed!
        pause
        goto menu
    ) else if "%option%"=="0" (
        echo.
        echo Goodbye! Deactivating Conda environment...
        call conda deactivate
        goto end
    ) else (
        echo Invalid option. Please try again.
        pause
        goto menu
    )
) else if "%choice%"=="7" (
    echo.
    echo Installing/Updating Dependencies...
    echo ======================================================

    REM Core packages via Conda
    echo Installing core packages via Conda...
    call conda install numpy opencv scikit-image pillow -y
    call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    call conda install jupyter pytest -y

    REM Additional packages via pip
    echo.
    echo Installing additional packages via pip...
    pip install --upgrade pip

    if exist "config\requirements.txt" (
        pip install -r config\requirements.txt
    ) else (
        REM Install minimal requirements if file not found
        pip install fastapi uvicorn pydantic
        pip install albumentations
        pip install tqdm loguru
        pip install matplotlib seaborn
    )

    echo.
    echo Installation completed!
    echo.
    conda list | findstr "torch numpy opencv"
    pause
) else (
    echo Invalid choice. Running Quick Demo...
    python simple_test.py
)

:end
echo.
echo ======================================================
echo Demo completed! Check output/ folder for results.
echo ======================================================
echo.
echo To reactivate the environment later, use:
echo   conda activate leap-PSW
echo.

REM Keep window open
pause