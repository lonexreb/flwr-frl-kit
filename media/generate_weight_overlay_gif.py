#!/usr/bin/env python3
"""Script to generate weight overlay animation as GIF."""

import subprocess
import os
import sys

def generate_gif(quality="medium"):
    """
    Generate the weight overlay animation as a GIF.

    Args:
        quality: "low", "medium", or "high" quality settings
    """

    # Quality settings
    settings = {
        "low": {
            "resolution": "480,270",
            "fps": "15"
        },
        "medium": {
            "resolution": "854,480",
            "fps": "20"
        },
        "high": {
            "resolution": "1280,720",
            "fps": "30"
        }
    }

    if quality not in settings:
        print(f"Invalid quality setting. Choose from: {list(settings.keys())}")
        return 1

    config = settings[quality]

    # Build the manim command using uv run
    cmd = [
        "uv", "run", "manim",
        "weight_overlay_animation.py",
        "WeightOverlayAnimation",
        "--format=gif",
        "-r", config["resolution"],
        "--fps", config["fps"]
    ]

    print(f"Generating weight overlay animation as GIF with {quality} quality...")
    print(f"Resolution: {config['resolution'].replace(',', 'x')}")
    print(f"FPS: {config['fps']}")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Change to scripts directory
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(scripts_dir)

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("\nGIF generated successfully!")
            print("Output should be in: media/videos/weight_overlay_animation/")

            # Try to find the generated GIF
            media_path = os.path.join(os.path.dirname(scripts_dir), "media", "videos", "weight_overlay_animation")
            if os.path.exists(media_path):
                for root, dirs, files in os.walk(media_path):
                    for file in files:
                        if file.endswith(".gif"):
                            gif_path = os.path.join(root, file)
                            print(f"Generated GIF: {gif_path}")
                            print(f"File size: {os.path.getsize(gif_path) / 1024:.2f} KB")
        else:
            print(f"\nError generating GIF:")
            print(result.stderr)
            return 1

    except FileNotFoundError:
        print("Error: uv or manim not found. Please install with: uv pip install manim")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    # Parse command line arguments
    quality = "medium"
    if len(sys.argv) > 1:
        quality = sys.argv[1].lower()

    sys.exit(generate_gif(quality))