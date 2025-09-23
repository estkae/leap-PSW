#!/bin/bash

# Linux/Mac shell script to run LFM2-VL Vision Recognition Demo

echo "======================================================"
echo "LFM2-VL Vision Recognition POC Demo Launcher"
echo "======================================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python version:"
python3 --version
echo

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "🔧 Virtual environment not found. Running setup..."
    echo
    python3 setup.py
    echo
    echo "✅ Setup completed. Please run this script again."
    exit 0
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if setup is complete
if [ ! -f "models/labels.txt" ]; then
    echo "📦 Models not found. Running setup..."
    python3 setup.py --skip-venv
fi

echo
echo "🎯 Available demo options:"
echo "1. Quick Demo (basic functionality)"
echo "2. Image Processing Demo"
echo "3. Batch Processing Demo"
echo "4. Model Optimization Demo"
echo "5. Performance Benchmark"
echo "6. Interactive Menu"
echo

read -p "Select demo (1-6): " choice

case $choice in
    1)
        echo "🚀 Running Quick Demo..."
        python3 quick_demo.py
        ;;
    2)
        echo "🖼️ Running Image Processing Demo..."
        python3 examples/demo.py --demo-type image --image-path data/test_images/test_face.jpg --verbose
        ;;
    3)
        echo "📦 Running Batch Processing Demo..."
        python3 examples/demo.py --demo-type batch --verbose
        ;;
    4)
        echo "⚡ Running Model Optimization Demo..."
        python3 examples/demo.py --demo-type optimize --target-device mobile --verbose
        ;;
    5)
        echo "🏁 Running Performance Benchmark..."
        python3 examples/demo.py --demo-type benchmark
        ;;
    6)
        # Interactive Menu
        while true; do
            clear
            echo "======================================================"
            echo "LFM2-VL Vision Recognition Interactive Menu"
            echo "======================================================"
            echo
            echo "1. Process single image"
            echo "2. Process batch of images"
            echo "3. Benchmark performance"
            echo "4. Test optimization"
            echo "5. Run unit tests"
            echo "6. View project info"
            echo "0. Exit"
            echo

            read -p "Select option (0-6): " option

            case $option in
                1)
                    echo
                    read -p "Enter image path (or press Enter for test image): " img_path
                    if [ -z "$img_path" ]; then
                        python3 examples/demo.py --demo-type image --image-path data/test_images/test_face.jpg --output-path output/result.jpg --verbose
                    else
                        python3 examples/demo.py --demo-type image --image-path "$img_path" --output-path output/result.jpg --verbose
                    fi
                    echo
                    echo "💾 Result saved to: output/result.jpg"
                    read -p "Press Enter to continue..."
                    ;;
                2)
                    python3 examples/demo.py --demo-type batch --verbose
                    read -p "Press Enter to continue..."
                    ;;
                3)
                    python3 examples/demo.py --demo-type benchmark
                    read -p "Press Enter to continue..."
                    ;;
                4)
                    echo
                    echo "📱 Available targets: mobile, ios, android, edge, tpu"
                    read -p "Enter target device: " target
                    python3 examples/demo.py --demo-type optimize --target-device "$target" --verbose
                    read -p "Press Enter to continue..."
                    ;;
                5)
                    echo "🧪 Running unit tests..."
                    python3 -m pytest tests/ -v
                    read -p "Press Enter to continue..."
                    ;;
                6)
                    echo
                    echo "======================================================"
                    echo "LFM2-VL Vision Recognition POC Information"
                    echo "======================================================"
                    echo
                    echo "📋 Version: 0.1.0 POC"
                    echo "🖥️ Platform: $(uname -s)"
                    echo "🐍 Python: $(python3 --version)"
                    echo
                    echo "📁 Project Structure:"
                    echo "   core/            - Vision processing modules"
                    echo "   examples/        - Demo scripts"
                    echo "   mobile/          - iOS/Android integration"
                    echo "   tests/           - Unit tests"
                    echo "   models/          - Mock model files"
                    echo "   data/            - Test images and data"
                    echo
                    echo "📚 For detailed documentation, see: README.md"
                    echo
                    read -p "Press Enter to continue..."
                    ;;
                0)
                    echo "👋 Goodbye!"
                    break
                    ;;
                *)
                    echo "❌ Invalid option. Please try again."
                    read -p "Press Enter to continue..."
                    ;;
            esac
        done
        ;;
    *)
        echo "❌ Invalid choice. Running Quick Demo..."
        python3 quick_demo.py
        ;;
esac

echo
echo "======================================================"
echo "✅ Demo completed! Check output/ folder for results."
echo "======================================================"

# On Mac, offer to open output folder
if [[ "$OSTYPE" == "darwin"* ]]; then
    read -p "Open output folder? (y/n): " open_folder
    if [[ $open_folder == "y" || $open_folder == "Y" ]]; then
        open output/ 2>/dev/null || echo "Output folder not found"
    fi
fi