#!/usr/bin/env python3
"""
Create real test images for LFM2-VL Vision Recognition Demo
Uses NumPy and OpenCV if available, falls back to PIL
"""

import os
import sys
from pathlib import Path

def create_with_numpy_cv2():
    """Create test images using NumPy and OpenCV"""
    import numpy as np
    import cv2

    print("Creating test images with OpenCV...")

    # Create directories
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test Face Image
    print("Creating test_face.jpg...")
    face_img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background

    # Face circle (skin tone)
    cv2.circle(face_img, (320, 240), 100, (180, 220, 255), -1)  # BGR format

    # Eyes
    cv2.circle(face_img, (290, 220), 15, (50, 50, 50), -1)  # Left eye
    cv2.circle(face_img, (350, 220), 15, (50, 50, 50), -1)  # Right eye

    # Eye pupils
    cv2.circle(face_img, (290, 220), 7, (0, 0, 0), -1)
    cv2.circle(face_img, (350, 220), 7, (0, 0, 0), -1)

    # Nose
    points = np.array([[320, 240], [310, 260], [330, 260]], np.int32)
    cv2.fillPoly(face_img, [points], (150, 180, 200))

    # Mouth
    cv2.ellipse(face_img, (320, 280), (30, 15), 0, 0, 180, (50, 50, 150), -1)

    # Add some text
    cv2.putText(face_img, "Test Face", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imwrite(str(test_dir / "test_face.jpg"), face_img)

    # Test Objects Image
    print("Creating test_objects.jpg...")
    objects_img = np.ones((480, 640, 3), dtype=np.uint8) * 180  # Gray background

    # Green rectangle
    cv2.rectangle(objects_img, (100, 100), (250, 200), (0, 255, 0), -1)
    cv2.putText(objects_img, "Object 1", (130, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Blue circle
    cv2.circle(objects_img, (450, 150), 60, (255, 0, 0), -1)
    cv2.putText(objects_img, "Object 2", (410, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Red triangle
    triangle = np.array([[320, 350], [250, 450], [390, 450]], np.int32)
    cv2.fillPoly(objects_img, [triangle], (0, 0, 255))
    cv2.putText(objects_img, "Object 3", (290, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Yellow square
    cv2.rectangle(objects_img, (500, 300), (600, 400), (0, 255, 255), -1)
    cv2.putText(objects_img, "Object 4", (515, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imwrite(str(test_dir / "test_objects.jpg"), objects_img)

    # Test Mixed Scene
    print("Creating test_mixed.jpg...")
    mixed_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Gradient background
    for y in range(480):
        for x in range(640):
            mixed_img[y, x] = [
                int(100 + (x / 640) * 100),  # Blue gradient
                int(50 + (y / 480) * 100),    # Green gradient
                100                            # Red constant
            ]

    # Multiple objects with transparency effect
    overlay = mixed_img.copy()

    # Person-like figure
    cv2.circle(overlay, (150, 100), 30, (180, 220, 255), -1)  # Head
    cv2.rectangle(overlay, (120, 130), (180, 250), (100, 150, 200), -1)  # Body

    # Car-like shape
    cv2.rectangle(overlay, (300, 200), (450, 280), (50, 50, 200), -1)  # Car body
    cv2.circle(overlay, (330, 280), 20, (30, 30, 30), -1)  # Wheel 1
    cv2.circle(overlay, (420, 280), 20, (30, 30, 30), -1)  # Wheel 2

    # Building
    cv2.rectangle(overlay, (500, 150), (600, 400), (150, 150, 150), -1)
    # Windows
    for i in range(3):
        for j in range(4):
            cv2.rectangle(overlay, (510 + i*30, 160 + j*50), (530 + i*30, 180 + j*50), (255, 255, 100), -1)

    # Tree
    cv2.circle(overlay, (250, 350), 40, (0, 150, 0), -1)  # Leaves
    cv2.rectangle(overlay, (245, 380), (255, 450), (50, 100, 150), -1)  # Trunk

    # Blend with original
    cv2.addWeighted(overlay, 0.7, mixed_img, 0.3, 0, mixed_img)

    # Add labels
    cv2.putText(mixed_img, "Person", (130, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(mixed_img, "Car", (350, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(mixed_img, "Building", (510, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(mixed_img, "Tree", (230, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(test_dir / "test_mixed.jpg"), mixed_img)

    print("Test images created successfully!")
    return True

def create_with_pil():
    """Create test images using PIL as fallback"""
    from PIL import Image, ImageDraw, ImageFont

    print("Creating test images with PIL...")

    # Create directories
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test Face Image
    print("Creating test_face.jpg...")
    face_img = Image.new('RGB', (640, 480), color=(200, 200, 200))
    draw = ImageDraw.Draw(face_img)

    # Face circle
    draw.ellipse([220, 140, 420, 340], fill=(255, 220, 180))

    # Eyes
    draw.ellipse([275, 205, 305, 235], fill=(50, 50, 50))  # Left eye
    draw.ellipse([335, 205, 365, 235], fill=(50, 50, 50))  # Right eye

    # Pupils
    draw.ellipse([283, 213, 297, 227], fill=(0, 0, 0))
    draw.ellipse([343, 213, 357, 227], fill=(0, 0, 0))

    # Nose
    draw.polygon([(320, 240), (310, 260), (330, 260)], fill=(200, 180, 150))

    # Mouth
    draw.arc([290, 270, 350, 290], start=0, end=180, fill=(150, 50, 50), width=3)

    # Text
    draw.text((250, 400), "Test Face", fill=(0, 0, 0))

    face_img.save(str(test_dir / "test_face.jpg"))

    # Test Objects Image
    print("Creating test_objects.jpg...")
    objects_img = Image.new('RGB', (640, 480), color=(180, 180, 180))
    draw = ImageDraw.Draw(objects_img)

    # Green rectangle
    draw.rectangle([100, 100, 250, 200], fill=(0, 255, 0))
    draw.text((130, 150), "Object 1", fill=(255, 255, 255))

    # Blue circle
    draw.ellipse([390, 90, 510, 210], fill=(0, 0, 255))
    draw.text((420, 145), "Object 2", fill=(255, 255, 255))

    # Red triangle
    draw.polygon([(320, 350), (250, 450), (390, 450)], fill=(255, 0, 0))
    draw.text((290, 410), "Object 3", fill=(255, 255, 255))

    # Yellow square
    draw.rectangle([500, 300, 600, 400], fill=(255, 255, 0))
    draw.text((515, 345), "Object 4", fill=(0, 0, 0))

    objects_img.save(str(test_dir / "test_objects.jpg"))

    # Test Mixed Scene
    print("Creating test_mixed.jpg...")
    mixed_img = Image.new('RGB', (640, 480), color=(150, 150, 150))
    draw = ImageDraw.Draw(mixed_img)

    # Gradient background (simplified)
    for i in range(0, 640, 20):
        color = (100 + i//6, 100, 150)
        draw.rectangle([i, 0, i+20, 480], fill=color)

    # Person-like figure
    draw.ellipse([135, 70, 165, 100], fill=(255, 220, 180))  # Head
    draw.rectangle([120, 100, 180, 220], fill=(100, 100, 200))  # Body
    draw.text((130, 60), "Person", fill=(255, 255, 255))

    # Car-like shape
    draw.rectangle([300, 200, 450, 280], fill=(200, 50, 50))  # Car body
    draw.ellipse([310, 260, 350, 300], fill=(30, 30, 30))  # Wheel 1
    draw.ellipse([400, 260, 440, 300], fill=(30, 30, 30))  # Wheel 2
    draw.text((350, 190), "Car", fill=(255, 255, 255))

    # Building
    draw.rectangle([500, 150, 600, 400], fill=(150, 150, 150))
    # Windows
    for i in range(3):
        for j in range(4):
            draw.rectangle([510 + i*30, 160 + j*50, 530 + i*30, 180 + j*50], fill=(255, 255, 100))
    draw.text((510, 140), "Building", fill=(255, 255, 255))

    # Tree
    draw.ellipse([210, 310, 290, 390], fill=(0, 150, 0))  # Leaves
    draw.rectangle([245, 380, 255, 450], fill=(100, 50, 0))  # Trunk
    draw.text((230, 300), "Tree", fill=(255, 255, 255))

    mixed_img.save(str(test_dir / "test_mixed.jpg"))

    print("Test images created successfully!")
    return True

def main():
    """Create test images using available libraries"""
    print("Creating real test images for LFM2-VL Vision Recognition Demo")
    print("-" * 60)

    # Try OpenCV first (better quality)
    try:
        import cv2
        import numpy as np
        print("Using OpenCV for image creation...")
        return create_with_numpy_cv2()
    except ImportError:
        print("OpenCV not available, trying PIL...")

    # Try PIL as fallback
    try:
        from PIL import Image, ImageDraw
        print("Using PIL for image creation...")
        return create_with_pil()
    except ImportError:
        print("PIL not available either.")

    print("\nERROR: Neither OpenCV nor PIL is available.")
    print("Please install one of them:")
    print("  conda install opencv")
    print("  OR")
    print("  pip install pillow")

    return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n" + "="*60)
        print("Success! Test images created in data/test_images/")
        print("You can now run the demo:")
        print("  python examples/demo.py --demo-type image")
    else:
        print("\nFailed to create test images.")
        sys.exit(1)