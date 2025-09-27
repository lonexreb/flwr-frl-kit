#!/usr/bin/env python3
"""Generate tree emoji PNGs with color filters for use in Manim animation."""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def create_emoji_png(emoji, filename, color_channel, size=200, emoji_font_size=109):
    """Create a PNG image of an emoji with color filter applied."""
    # Create a transparent image
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to find Noto Color Emoji font
    font_paths = [
        "NotoColorEmoji-Regular.ttf",  # Local font file (first priority)
        "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",  # Linux
        "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
        "C:/Windows/Fonts/seguiemj.ttf",  # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Fallback
    ]

    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, size=emoji_font_size)
                print(f"Successfully loaded font: {font_path} with size {emoji_font_size}")
                break
            except Exception as e:
                print(f"Failed to load {font_path}: {e}")
                continue

    if font is None:
        print("Warning: No suitable emoji font found, using default")
        font = ImageFont.load_default()

    # Draw the emoji
    bbox = draw.textbbox((0, 0), emoji, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2

    # Draw emoji in white first to get the shape
    draw.text((x, y), emoji, font=font, fill=(255, 255, 255, 255))

    # Convert to numpy array for color filtering
    img_array = np.array(img)

    # Create color filter based on channel
    if color_channel == 'R':
        # Keep only red channel, zero out green and blue
        img_array[:, :, 1] = 0  # Green
        img_array[:, :, 2] = 0  # Blue
    elif color_channel == 'G':
        # Keep only green channel, zero out red and blue
        img_array[:, :, 0] = 0  # Red
        img_array[:, :, 2] = 0  # Blue
    elif color_channel == 'B':
        # Keep only blue channel, zero out red and green
        img_array[:, :, 0] = 0  # Red
        img_array[:, :, 1] = 0  # Green

    # Convert back to PIL Image
    filtered_img = Image.fromarray(img_array)

    # Save the image
    filtered_img.save(filename)
    print(f"Created {filename} with {color_channel} channel filter")

def main():
    """Generate device emoji PNGs with different color channels."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create device emoji PNGs with single color channels using proper emoji font size
    create_emoji_png("🤖", os.path.join(script_dir, "robot_red.png"), 'R', size=200, emoji_font_size=109)
    create_emoji_png("💻", os.path.join(script_dir, "laptop_green.png"), 'G', size=200, emoji_font_size=109)
    create_emoji_png("📱", os.path.join(script_dir, "phone_blue.png"), 'B', size=200, emoji_font_size=109)

    # Also create natural colored versions
    create_emoji_png("🤖", os.path.join(script_dir, "robot.png"), 'G', size=200, emoji_font_size=109)
    create_emoji_png("💻", os.path.join(script_dir, "laptop.png"), 'G', size=200, emoji_font_size=109)
    create_emoji_png("📱", os.path.join(script_dir, "phone.png"), 'G', size=200, emoji_font_size=109)

    # Create lock emoji
    create_emoji_png("🔒", os.path.join(script_dir, "lock.png"), 'G', size=200, emoji_font_size=109)

if __name__ == "__main__":
    main()