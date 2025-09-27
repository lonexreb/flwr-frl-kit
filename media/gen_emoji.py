from PIL import Image, ImageDraw, ImageFont
import os

# --- Configuration for Single Emoji on Transparent Background (Linux/Ubuntu) ---
EMOJI = "🔒"  # The single emoji you want to generate
FONT_PATH = "NotoColorEmoji-Regular.ttf"  # Local font file 
FONT_SIZE = 109 # Larger size for a single, focused emoji
OUTPUT_FILENAME = "lock.png"
CANVAS_SIZE = (150, 150) # Canvas size slightly larger than the font size

# --- Script ---

# 1. Check if the font file exists
if not os.path.exists(FONT_PATH):
    print(f"Error: Font file not found at {FONT_PATH}.")
    print("Please ensure the 'fonts-noto-color-emoji' package is installed on your Linux system.")
    print("You may need to run: sudo apt install fonts-noto-color-emoji")
    exit(1)

# 2. Load the font
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError as e:
    print(f"Error loading font with Pillow: {e}")
    exit(1)

# 3. Create an RGBA (transparent) canvas
# (0, 0, 0, 0) means Red=0, Green=0, Blue=0, Alpha=0 (fully transparent).
img = Image.new('RGBA', CANVAS_SIZE, (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# 4. Calculate text position to center the emoji precisely
# Get the bounding box of the text.
bbox = draw.textbbox((0, 0), EMOJI, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

# Center the text horizontally
x = (CANVAS_SIZE[0] - text_width) / 2
# Center the text vertically (adjusting for the baseline offset)
y = (CANVAS_SIZE[1] - text_height) / 2 - bbox[1]

# 5. Draw the emoji using the embedded_color=True option
# This ensures the emoji renders in its full color from the font file.
draw.text(
    (x, y), 
    EMOJI, 
    fill="white", # Fill color is required but usually ignored for color glyphs
    embedded_color=True, 
    font=font
)

# 6. Save the image as a PNG with transparency.
img.save(OUTPUT_FILENAME, "PNG")

print(f"Successfully rendered single emoji '{EMOJI}' with a transparent background.")
print(f"Saved as {os.path.abspath(OUTPUT_FILENAME)}.")