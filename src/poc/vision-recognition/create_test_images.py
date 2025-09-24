#!/usr/bin/env python3
"""
Create test images for LFM2-VL Vision Recognition Demo
Creates synthetic test images without external dependencies
"""

import os
from pathlib import Path


def create_ppm_image(width, height, pixels):
    """Create a PPM image (simple format, no external libs needed)"""
    header = f"P3\n{width} {height}\n255\n"
    pixel_data = []

    for row in pixels:
        for r, g, b in row:
            pixel_data.append(f"{r} {g} {b}")

    return header + " ".join(pixel_data)


def create_test_face_image():
    """Create a simple face-like test image"""
    width, height = 100, 100
    pixels = []

    # Create face-colored background (skin tone)
    bg_color = (255, 220, 180)

    for y in range(height):
        row = []
        for x in range(width):
            # Default background
            r, g, b = bg_color

            # Draw face circle
            cx, cy = 50, 50
            radius = 40
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

            if dist < radius:
                # Inside face
                r, g, b = (255, 220, 180)

                # Left eye
                if 30 < x < 40 and 40 < y < 45:
                    r, g, b = (50, 50, 50)

                # Right eye
                if 60 < x < 70 and 40 < y < 45:
                    r, g, b = (50, 50, 50)

                # Mouth
                if 40 < x < 60 and 65 < y < 70:
                    r, g, b = (150, 50, 50)
            else:
                # Outside face - background
                r, g, b = (200, 200, 200)

            row.append((r, g, b))
        pixels.append(row)

    return create_ppm_image(width, height, pixels)


def create_test_objects_image():
    """Create a simple objects test image"""
    width, height = 100, 100
    pixels = []

    for y in range(height):
        row = []
        for x in range(width):
            # Default gray background
            r, g, b = (180, 180, 180)

            # Green rectangle (object 1)
            if 20 < x < 40 and 20 < y < 40:
                r, g, b = (50, 200, 50)

            # Blue circle (object 2)
            cx, cy = 70, 30
            radius = 15
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist < radius:
                r, g, b = (50, 50, 200)

            # Red triangle area (object 3)
            if 30 < x < 70 and 60 < y < 80:
                if x - 30 < y - 60:
                    r, g, b = (200, 50, 50)

            row.append((r, g, b))
        pixels.append(row)

    return create_ppm_image(width, height, pixels)


def create_test_mixed_image():
    """Create a mixed scene test image"""
    width, height = 100, 100
    pixels = []

    for y in range(height):
        row = []
        for x in range(width):
            # Gradient background
            r = int(100 + (x / width) * 100)
            g = int(100 + (y / height) * 100)
            b = 150

            # Multiple objects
            # Yellow square
            if 10 < x < 25 and 10 < y < 25:
                r, g, b = (255, 255, 100)

            # Cyan rectangle
            if 40 < x < 70 and 20 < y < 35:
                r, g, b = (100, 255, 255)

            # Magenta circle
            cx, cy = 30, 70
            radius = 12
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist < radius:
                r, g, b = (255, 100, 255)

            # Orange triangle
            if 60 < x < 90 and 60 < y < 90:
                if x - 60 > (90 - y):
                    r, g, b = (255, 150, 50)

            row.append((r, g, b))
        pixels.append(row)

    return create_ppm_image(width, height, pixels)


def ppm_to_jpg_header():
    """Create a minimal JPEG file header (simplified)"""
    # This is a very simplified approach - in practice, we'd use PIL or cv2
    # For now, we'll just create placeholder files
    return b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'


def save_test_image(filename, image_data):
    """Save test image as a simple format"""
    # For simplicity, save as .ppm (portable pixmap) which can be read by most tools
    # Then create a .jpg placeholder

    # Save as PPM
    ppm_path = filename.replace('.jpg', '.ppm')
    with open(ppm_path, 'w') as f:
        f.write(image_data)

    # Create a minimal JPG placeholder (won't be a valid image, but file will exist)
    with open(filename, 'wb') as f:
        f.write(ppm_to_jpg_header())
        f.write(b'\xFF\xD9')  # End of JPEG marker

    return filename


def main():
    """Create all test images"""
    print("Creating test images for LFM2-VL Vision Recognition Demo")
    print("-" * 50)

    # Create directories
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test images
    images = [
        ("test_face.jpg", create_test_face_image()),
        ("test_objects.jpg", create_test_objects_image()),
        ("test_mixed.jpg", create_test_mixed_image())
    ]

    for filename, image_data in images:
        filepath = test_dir / filename

        # Save PPM version (readable)
        ppm_path = str(filepath).replace('.jpg', '.ppm')
        with open(ppm_path, 'w') as f:
            f.write(image_data)
        print(f"Created: {ppm_path}")

        # Create JPG placeholder
        with open(filepath, 'wb') as f:
            # Write minimal JPEG structure
            f.write(b'\xFF\xD8\xFF\xE0')  # SOI and APP0 marker
            f.write(b'\x00\x10')  # Length
            f.write(b'JFIF\x00')  # Identifier
            f.write(b'\x01\x01')  # Version
            f.write(b'\x00')  # Aspect ratio units
            f.write(b'\x00\x01\x00\x01')  # X and Y density
            f.write(b'\x00\x00')  # Thumbnail size
            f.write(b'\xFF\xD9')  # EOI marker
        print(f"Created: {filepath}")

    # Create description files
    descriptions = [
        ("test_face.txt", "Synthetic face image with eyes and mouth features"),
        ("test_objects.txt", "Multiple colored objects: green rectangle, blue circle, red triangle"),
        ("test_mixed.txt", "Mixed scene with various shapes and gradient background")
    ]

    for filename, description in descriptions:
        filepath = test_dir / filename
        with open(filepath, 'w') as f:
            f.write(description)
        print(f"Created: {filepath}")

    print("-" * 50)
    print(f"Test images created in: {test_dir}")
    print("\nNote: PPM files contain actual image data")
    print("      JPG files are placeholders for compatibility")
    print("\nTo view PPM images, use:")
    print("  - Any image viewer that supports PPM format")
    print("  - Convert to JPG with: ImageMagick, GIMP, or Python PIL")

    return True


if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)