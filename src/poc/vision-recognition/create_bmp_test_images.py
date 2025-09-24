#!/usr/bin/env python3
"""
Create BMP test images using only Python standard library
BMP is a simple format that can be created without external dependencies
"""

import struct
import os
from pathlib import Path


def create_bmp_header(width, height):
    """Create BMP file header"""
    file_size = 54 + width * height * 3  # Header + pixel data
    reserved = 0
    offset = 54

    # BMP Header
    header = b'BM'  # Signature
    header += struct.pack('<I', file_size)  # File size
    header += struct.pack('<H', reserved)   # Reserved 1
    header += struct.pack('<H', reserved)   # Reserved 2
    header += struct.pack('<I', offset)     # Pixel data offset

    # DIB Header (BITMAPINFOHEADER)
    header += struct.pack('<I', 40)         # Header size
    header += struct.pack('<I', width)      # Image width
    header += struct.pack('<I', height)     # Image height
    header += struct.pack('<H', 1)          # Color planes
    header += struct.pack('<H', 24)         # Bits per pixel
    header += struct.pack('<I', 0)          # Compression (none)
    header += struct.pack('<I', width * height * 3)  # Image size
    header += struct.pack('<I', 2835)       # X pixels per meter
    header += struct.pack('<I', 2835)       # Y pixels per meter
    header += struct.pack('<I', 0)          # Colors in palette
    header += struct.pack('<I', 0)          # Important colors

    return header


def create_test_face_bmp():
    """Create a simple face BMP image"""
    width, height = 200, 200
    pixels = []

    for y in range(height):
        for x in range(width):
            # Background (light gray)
            r, g, b = 200, 200, 200

            # Face circle (center at 100, 100, radius 80)
            cx, cy, radius = 100, 100, 80
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

            if dist < radius:
                # Face color (skin tone)
                r, g, b = 255, 220, 180

                # Left eye (circle at 75, 85, radius 8)
                eye_dist = ((x - 75) ** 2 + (y - 85) ** 2) ** 0.5
                if eye_dist < 8:
                    r, g, b = 50, 50, 50
                    if eye_dist < 4:  # Pupil
                        r, g, b = 0, 0, 0

                # Right eye (circle at 125, 85, radius 8)
                eye_dist = ((x - 125) ** 2 + (y - 85) ** 2) ** 0.5
                if eye_dist < 8:
                    r, g, b = 50, 50, 50
                    if eye_dist < 4:  # Pupil
                        r, g, b = 0, 0, 0

                # Mouth (rectangle)
                if 80 < x < 120 and 125 < y < 135:
                    r, g, b = 150, 50, 50

                # Nose (small triangle area)
                if 95 < x < 105 and 100 < y < 110:
                    r, g, b = 200, 180, 150

            # BMP stores pixels in BGR order, bottom-up
            pixels.append((b, g, r))

    # Reverse rows for bottom-up storage
    pixel_data = []
    for row in range(height - 1, -1, -1):
        row_start = row * width
        row_end = row_start + width
        pixel_data.extend(pixels[row_start:row_end])

    return width, height, pixel_data


def create_test_objects_bmp():
    """Create a simple objects BMP image"""
    width, height = 200, 200
    pixels = []

    for y in range(height):
        for x in range(width):
            # Background (gray)
            r, g, b = 180, 180, 180

            # Green rectangle (top-left)
            if 20 < x < 80 and 20 < y < 60:
                r, g, b = 0, 200, 0

            # Blue circle (top-right)
            cx, cy, radius = 140, 40, 25
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist < radius:
                r, g, b = 0, 0, 200

            # Red square (bottom-left)
            if 20 < x < 80 and 120 < y < 180:
                r, g, b = 200, 0, 0

            # Yellow circle (bottom-right)
            cx, cy, radius = 140, 150, 25
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist < radius:
                r, g, b = 255, 255, 0

            # Purple triangle (center) - simplified as diamond
            cx, cy = 100, 100
            if abs(x - cx) + abs(y - cy) < 30:
                r, g, b = 200, 0, 200

            pixels.append((b, g, r))

    # Reverse rows for bottom-up storage
    pixel_data = []
    for row in range(height - 1, -1, -1):
        row_start = row * width
        row_end = row_start + width
        pixel_data.extend(pixels[row_start:row_end])

    return width, height, pixel_data


def create_test_mixed_bmp():
    """Create a mixed scene BMP image"""
    width, height = 200, 200
    pixels = []

    for y in range(height):
        for x in range(width):
            # Gradient background
            r = 100 + int((x / width) * 100)
            g = 100 + int((y / height) * 100)
            b = 150

            # Person-like figure (left)
            if 30 < x < 70:
                # Head
                cx, cy, radius = 50, 40, 15
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                if dist < radius:
                    r, g, b = 255, 220, 180

                # Body
                if 40 < x < 60 and 55 < y < 100:
                    r, g, b = 100, 100, 200

            # Car-like shape (right)
            if 120 < x < 180 and 80 < y < 110:
                r, g, b = 200, 50, 50
                # Windows
                if 130 < x < 150 and 85 < y < 95:
                    r, g, b = 100, 150, 200

            # Building (center)
            if 85 < x < 115 and 120 < y < 180:
                r, g, b = 150, 150, 150
                # Windows (grid pattern)
                if ((x - 85) % 10 < 8) and ((y - 120) % 15 < 10):
                    if ((x - 85) // 10 + (y - 120) // 15) % 2 == 0:
                        r, g, b = 255, 255, 100

            pixels.append((b, g, r))

    # Reverse rows for bottom-up storage
    pixel_data = []
    for row in range(height - 1, -1, -1):
        row_start = row * width
        row_end = row_start + width
        pixel_data.extend(pixels[row_start:row_end])

    return width, height, pixel_data


def save_bmp(filename, width, height, pixel_data):
    """Save pixel data as BMP file"""
    with open(filename, 'wb') as f:
        # Write header
        f.write(create_bmp_header(width, height))

        # Write pixel data
        for pixel in pixel_data:
            f.write(struct.pack('BBB', *pixel))

    print(f"Created: {filename}")


def main():
    """Create BMP test images"""
    print("Creating BMP test images for LFM2-VL Vision Recognition Demo")
    print("-" * 60)

    # Create directory
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test images
    images = [
        ("test_face.bmp", create_test_face_bmp()),
        ("test_objects.bmp", create_test_objects_bmp()),
        ("test_mixed.bmp", create_test_mixed_bmp())
    ]

    for filename, (width, height, pixel_data) in images:
        filepath = test_dir / filename
        save_bmp(str(filepath), width, height, pixel_data)

    print("-" * 60)
    print("BMP test images created successfully!")
    print(f"Location: {test_dir}")
    print("\nBMP files can be:")
    print("  - Opened with any image viewer")
    print("  - Read by OpenCV when available")
    print("  - Converted to JPG using online tools")

    # Create info file
    info_file = test_dir / "image_info.txt"
    with open(info_file, 'w') as f:
        f.write("Test Images for LFM2-VL Vision Recognition POC\n")
        f.write("=" * 50 + "\n\n")
        f.write("test_face.bmp    - Synthetic face with eyes and mouth\n")
        f.write("test_objects.bmp - Multiple colored shapes and objects\n")
        f.write("test_mixed.bmp   - Complex scene with person, car, building\n")
        f.write("\nFormat: BMP (Bitmap) - 24-bit RGB\n")
        f.write("Size: 200x200 pixels\n")
        f.write("\nThese images work without any external libraries.\n")

    print(f"\nInfo file created: {info_file}")

    return True


if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)