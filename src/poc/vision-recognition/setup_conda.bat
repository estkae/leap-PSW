@echo off
REM Setup script for LEAP-PSW Vision Recognition POC with Conda

echo ======================================================
echo LEAP-PSW Vision Recognition - Conda Setup
echo ======================================================
echo.

REM Check for Conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH
    echo.
    echo Please install Anaconda or Miniconda:
    echo   - Anaconda: https://www.anaconda.com/products/individual
    echo   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo Found Conda installation:
conda --version
echo.

REM Check if environment already exists
echo Checking for existing leap-PSW environment...
conda env list | findstr /C:"leap-PSW" >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo Environment 'leap-PSW' already exists!
    echo.
    set /p update="Do you want to update it? (y/n): "
    if /i "%update%"=="y" (
        echo Updating existing environment...
        call conda env update -f environment.yml
    ) else (
        echo Using existing environment without updates.
    )
) else (
    echo.
    echo Creating new Conda environment 'leap-PSW'...
    echo This may take several minutes...
    echo.

    REM Create environment from YAML file if exists
    if exist "environment.yml" (
        echo Using environment.yml file...
        call conda env create -f environment.yml
    ) else (
        echo environment.yml not found. Creating basic environment...
        call conda create -n leap-PSW python=3.10 -y

        echo.
        echo Installing packages...
        call conda activate leap-PSW

        REM Install core packages
        call conda install numpy opencv scikit-image pillow matplotlib seaborn -y
        call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        call conda install jupyter pytest pandas h5py -y

        REM Install pip packages
        pip install tflite-runtime albumentations fastapi uvicorn loguru
        pip install onnx onnxruntime coremltools psutil

        call conda deactivate
    )

    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Environment creation failed!
        pause
        exit /b 1
    )
)

REM Activate environment
echo.
echo Activating leap-PSW environment...
call conda activate leap-PSW

if %errorlevel% neq 0 (
    echo ERROR: Failed to activate environment
    pause
    exit /b 1
)

echo Environment activated successfully!
echo.
python --version
echo.

REM Create project structure
echo Setting up project structure...

if not exist "models" mkdir models
if not exist "data\test_images" mkdir data\test_images
if not exist "output" mkdir output
if not exist "logs" mkdir logs
if not exist "cache" mkdir cache

REM Run quick setup to create mock files
if exist "quick_setup.py" (
    echo Creating mock models and test data...
    python quick_setup.py
) else (
    echo Warning: quick_setup.py not found. Skipping mock file creation.
)

REM Test installation
echo.
echo ======================================================
echo Testing Installation...
echo ======================================================

REM Test imports
python -c "import numpy; print(f'NumPy {numpy.__version__} - OK')" 2>nul || echo NumPy - FAILED
python -c "import cv2; print(f'OpenCV {cv2.__version__} - OK')" 2>nul || echo OpenCV - FAILED
python -c "import torch; print(f'PyTorch {torch.__version__} - OK')" 2>nul || echo PyTorch - FAILED
python -c "import PIL; print(f'Pillow {PIL.__version__} - OK')" 2>nul || echo Pillow - FAILED

REM Run simple test if available
echo.
if exist "simple_test.py" (
    echo Running functionality test...
    echo ------------------------------------------------------
    python simple_test.py
) else (
    echo simple_test.py not found. Skipping functionality test.
)

echo.
echo ======================================================
echo Setup Complete!
echo ======================================================
echo.
echo Environment: leap-PSW
echo Location: %CONDA_PREFIX%
echo.
echo To use this environment:
echo   1. Activate: conda activate leap-PSW
echo   2. Run demo: python simple_test.py
echo   3. Or use:  run_demo_conda.bat
echo.
echo To run the full demo:
echo   run_demo_conda.bat
echo.
echo To deactivate when done:
echo   conda deactivate
echo.

pause