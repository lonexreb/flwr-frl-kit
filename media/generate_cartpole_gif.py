#!/usr/bin/env python3
"""Generate a GIF of a single CartPole run from gymnasium."""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packages.rl_core.envs.make_env import make_env


def generate_cartpole_gif(
    output_path: str = "cartpole_run.gif",
    env_id: str = "CartPole-v1",
    seed: int = 42,
    max_steps: int = 500,
    fps: int = 30,
    black_background: bool = False,
    square_crop: bool = True
):
    """Generate a GIF of a single CartPole run with random actions.

    Args:
        output_path: Path where the GIF will be saved
        env_id: Environment ID (default: CartPole-v1)
        seed: Random seed for reproducibility
        max_steps: Maximum number of steps to record
        fps: Frames per second for the GIF
        black_background: Convert white background to black
        square_crop: Crop to square around center
    """
    # Create environment with rendering
    env = gym.make(env_id, render_mode="rgb_array")

    # Modify termination conditions to allow pole to reach 90 degrees
    # Default is ~12 degrees (0.209 radians), we'll set to 90 degrees (1.57 radians)
    env.unwrapped.theta_threshold_radians = np.pi / 2  # 90 degrees

    # Also increase cart position threshold to allow more movement
    env.unwrapped.x_threshold = 10.0  # Default is 2.4

    # Increase pendulum length (default is 0.5)
    env.unwrapped.length = 2.0  # Quadruple the pendulum length

    print(f"Modified environment parameters:")
    print(f"  Max pole angle: ±{env.unwrapped.theta_threshold_radians:.3f} rad (±{env.unwrapped.theta_threshold_radians * 180 / np.pi:.1f}°)")
    print(f"  Max cart position: ±{env.unwrapped.x_threshold}")
    print(f"  Pendulum length: {env.unwrapped.length}")

    frames = []
    observation, info = env.reset(seed=seed)

    print(f"Recording CartPole run with seed {seed}...")

    for step in range(max_steps):
        # Take random action
        action = env.action_space.sample()

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Debug info
        if terminated:
            print(f"  Step {step+1}: Terminated - pole angle: {observation[2]:.3f} rad ({observation[2] * 180 / np.pi:.1f}°), cart pos: {observation[0]:.3f}")
        if truncated:
            print(f"  Step {step+1}: Truncated after max steps")

        # Capture frame
        frame = env.render()
        if frame is not None:
            # Convert to PIL Image
            img = Image.fromarray(frame)

            # Convert white background to black if requested
            if black_background:
                # Convert to numpy array for processing
                img_array = np.array(img)

                # Define white color range (close to white due to anti-aliasing)
                white_mask = np.all(img_array >= [240, 240, 240], axis=2)

                # Replace white pixels with black
                img_array[white_mask] = [0, 0, 0]

                # Convert back to PIL Image
                img = Image.fromarray(img_array)

            # Square crop around center if requested
            if square_crop:
                width, height = img.size
                min_dim = min(width, height)

                # Calculate crop box for center square
                left = (width - min_dim) // 2
                top = (height - min_dim) // 2
                right = left + min_dim
                bottom = top + min_dim

                img = img.crop((left, top, right, bottom))

                # Resize to 300x300
                img = img.resize((300, 300), Image.LANCZOS)

            frames.append(img)

        # Check if episode ended
        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps")
            break

    env.close()

    if frames:
        # Save as GIF
        duration = 1000 // fps  # Duration per frame in milliseconds
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to: {output_path}")
        print(f"Total frames: {len(frames)}")
    else:
        print("No frames captured!")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate a GIF of a CartPole run")

    parser.add_argument(
        "--output",
        type=str,
        default="cartpole_run.gif",
        help="Output path for the GIF (default: cartpole_run.gif)"
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="CartPole-v1",
        help="Environment ID (default: CartPole-v1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum number of steps (default: 500)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the GIF (default: 30)"
    )
    parser.add_argument(
        "--black-background",
        action="store_true",
        help="Convert white background to black (default: False)"
    )
    parser.add_argument(
        "--square-crop",
        action="store_true",
        help="Crop to square around center (default: False)"
    )
    parser.add_argument(
        "--generate-multiple",
        action="store_true",
        help="Generate 3 GIFs with different initial conditions (default: False)"
    )

    args = parser.parse_args()

    # Always generate 3 GIFs with different seeds/initial conditions by default
    seeds = [42, 123, 456]
    base_name = args.output.replace('.gif', '')

    print("Generating 3 GIFs with different initial conditions...")

    for i, seed in enumerate(seeds, 1):
        output_path = f"{base_name}_{i}.gif"
        print(f"\n=== Generating GIF {i}/3 (seed: {seed}) ===")

        generate_cartpole_gif(
            output_path=output_path,
            env_id=args.env_id,
            seed=seed,
            max_steps=args.max_steps,
            fps=args.fps,
            black_background=args.black_background,
            square_crop=True  # Always crop to square
        )

    print(f"\n✓ Generated 3 GIFs: {base_name}_1.gif, {base_name}_2.gif, {base_name}_3.gif")


if __name__ == "__main__":
    main()