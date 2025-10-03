# Media Generation Scripts

This directory contains scripts for generating animations and visualizations for federated reinforcement learning demonstrations.

## Contents

### Animation Scripts

- **`weight_overlay_animation.py`** - Manim animation showing federated averaging of weight matrices from 3 clients
- **`weight_overlay_animation_complex.py`** - Extended version with client updates and global model convergence
- **`generate_cartpole_gif.py`** - Generate GIF animations of CartPole gymnasium environment runs

### Asset Generation

- **`generate_tree_emojis.py`** - Creates colored tree emoji PNGs (red, green, blue) for client visualization
- **`gen_emoji.py`** - General emoji PNG generation with color filters
- **`generate_weight_overlay_gif.py`** - Helper script for weight overlay GIF generation

### Assets

- **Images**: Tree emojis, device emojis (phone, laptop, robot), lock icons
- **Font**: `NotoColorEmoji-Regular.ttf` - Emoji font for rendering
- **Output**: Generated GIFs and videos in subdirectories (`images/`, `videos/`, `media/`)

## Usage

### Generate Weight Overlay Animation

```bash
python weight_overlay_animation.py
```

This creates an MP4 video at 60fps (default high quality mode) showing:
1. Three clients with devices (phone, robot, laptop)
2. Client weight matrices in different colors (blue, red, green)
3. Secure aggregation with lock icons
4. Final aggregated global model

### Generate CartPole GIF

```bash
python generate_cartpole_gif.py --output cartpole_run.gif --seed 42
```

Options:
- `--env-id` - Gymnasium environment (default: CartPole-v1)
- `--seed` - Random seed for reproducibility
- `--max-steps` - Maximum episode steps (default: 500)
- `--fps` - Frames per second (default: 30)
- `--black-background` - Convert white background to black
- `--square-crop` - Crop to square format

### Generate Emoji Assets

```bash
python generate_tree_emojis.py
```

Creates colored tree PNGs for client visualization.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
```

## Output Directories

- `videos/` - Generated video files
- `images/` - Generated image frames
- `media/` - Manim output directory
- `Tex/` - LaTeX rendering cache (Manim)

## Notes

- Manim animations use the Manim Community Edition library
- CartPole scripts require gymnasium for RL environment simulation
- Emoji generation requires NotoColorEmoji font file in the same directory
