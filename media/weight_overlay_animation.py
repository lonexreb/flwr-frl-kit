#!/usr/bin/env python3
"""Manim animation showing 3 weight matrices being overlayed to create a final weight."""

from manim import *
import numpy as np
import os


class WeightOverlayAnimation(Scene):
    def construct(self):
        # Set black background
        self.camera.background_color = BLACK

        # Create 3 weight matrices (represented as grids with colors)
        weight1_data = np.random.rand(4, 4) * 0.8 + 0.1  # Client 1 weights
        weight2_data = np.random.rand(4, 4) * 0.8 + 0.1  # Client 2 weights
        weight3_data = np.random.rand(4, 4) * 0.8 + 0.1  # Client 3 weights

        # Create visual representations of weight matrices
        def create_weight_matrix(data, label, color_scheme):
            # Create a grid of squares representing the matrix
            matrix_group = VGroup()

            for i in range(4):
                for j in range(4):
                    # Create square with color intensity based on weight value
                    intensity = data[i, j]
                    square = Square(side_length=0.4)
                    square.set_fill(color_scheme, opacity=intensity)
                    square.set_stroke(WHITE, width=1)
                    square.move_to([j * 0.5 - 0.75, -i * 0.5 + 0.75, 0])
                    matrix_group.add(square)

            # Add label
            label_text = Text(label, font_size=24)
            label_text.next_to(matrix_group, DOWN, buff=0.1)
            matrix_group.add(label_text)

            return matrix_group

        # Create tree image representations using standalone images
        def create_tree_image(weight_data, color_scheme, label, tree_filename):
            # Calculate average weight to determine tree opacity
            avg_weight = np.mean(weight_data)

            # Load standalone colored tree PNG
            tree_png_path = tree_filename

            try:
                # Create tree image
                tree_img = ImageMobject(tree_png_path)
                tree_img.scale(0.7)  # Larger tree scale

                # Apply opacity based on weight
                tree_img.set_opacity(0.4 + 0.6 * avg_weight)

                # Add client label below
                label_text = Text(label, font_size=24, color=color_scheme)
                label_text.next_to(tree_img, DOWN, buff=0.3)

                tree_group = Group(tree_img, label_text)
                return tree_group
            except Exception as e:
                print(f"Error loading {tree_filename}: {e}")
                # Fallback: return just the label text
                label_text = Text(label, font_size=24, color=color_scheme)
                return Group(label_text)

        # Create tree images for each client underneath where weights will appear (moved even further left)
        tree1 = create_tree_image(weight1_data, BLUE, "Client 1", "blue_tree.png")
        tree1.move_to([-4.5, 2.5, 0])  # Same position as weight1

        tree2 = create_tree_image(weight2_data, RED, "Client 2", "red_tree.png")
        tree2.move_to([-4.5, 0, 0])  # Same position as weight2

        tree3 = create_tree_image(weight3_data, GREEN, "Client 3", "green_tree.png")
        tree3.move_to([-4.5, -2.5, 0])  # Same position as weight3

        # Create device emojis to the left of weight matrices (uncolored versions) - larger and further left
        device1 = create_tree_image(weight1_data, WHITE, "", "phone.png")
        device1[0].scale(1.5)  # Make device larger
        device1.move_to([-6.5, 2.5, 0])

        device2 = create_tree_image(weight2_data, WHITE, "", "robot.png")
        device2[0].scale(1.5)  # Make device larger
        device2.move_to([-6.5, 0, 0])

        device3 = create_tree_image(weight3_data, WHITE, "", "laptop.png")
        device3[0].scale(1.5)  # Make device larger
        device3.move_to([-6.5, -2.5, 0])

        # Create the three weight matrices after trees (moved even further left)
        weight1 = create_weight_matrix(weight1_data, "Client 1", BLUE)
        weight2 = create_weight_matrix(weight2_data, "Client 2", RED)
        weight3 = create_weight_matrix(weight3_data, "Client 3", GREEN)

        # Stack matrices even further left
        weight1.move_to([-4.5, 2.5, 0])
        weight2.move_to([-4.5, 0, 0])
        weight3.move_to([-4.5, -2.5, 0])

        # Show the tree emojis and device emojis first
        self.play(
            FadeIn(tree1),
            FadeIn(tree2),
            FadeIn(tree3),
            FadeIn(device1),
            FadeIn(device2),
            FadeIn(device3),
            run_time=2
        )
        self.wait(2)  # Let the trees and devices be visible for a bit

        # Fade out trees and fade in weight matrices in the same positions (keep devices visible)
        self.play(
            FadeOut(tree1),
            FadeOut(tree2),
            FadeOut(tree3),
            FadeIn(weight1),
            FadeIn(weight2),
            FadeIn(weight3),
            run_time=2
        )

        self.wait(1)

        # Create connecting lines with 45-degree curved elbows and equal segment lengths
        # Calculate equal segment length for 45-degree path (extended for new weight position)
        segment_length = 1.5

        # Top line: goes right then down at 45 degrees
        arrow1 = VGroup()
        start1 = weight1.get_right() + RIGHT * 0.1
        mid1 = [start1[0] + segment_length, start1[1], 0]
        end1 = [mid1[0] + segment_length, mid1[1] - segment_length, 0]
        line1_h = Line(start1, mid1, color=BLUE, stroke_width=3)
        line1_v = Line(mid1, end1, color=BLUE, stroke_width=3)
        arrow1.add(line1_h, line1_v)

        # Middle line: goes right (unchanged)
        arrow2 = Line(
            weight2.get_right() + RIGHT * 0.1,
            [weight2.get_right()[0] + 2 * segment_length + 0.1, 0, 0],
            color=RED,
            stroke_width=3
        )

        # Bottom line: goes right then up at 45 degrees
        arrow3 = VGroup()
        start3 = weight3.get_right() + RIGHT * 0.1
        mid3 = [start3[0] + segment_length, start3[1], 0]
        end3 = [mid3[0] + segment_length, mid3[1] + segment_length, 0]
        line3_h = Line(start3, mid3, color=GREEN, stroke_width=3)
        line3_v = Line(mid3, end3, color=GREEN, stroke_width=3)
        arrow3.add(line3_h, line3_v)

        self.play(
            Create(arrow1),
            Create(arrow2),
            Create(arrow3),
            run_time=1.5
        )

        # Create lock emojis at inflection points and on middle line
        # Lock for arrow1 inflection point (new calculated position) - larger size
        lock1 = ImageMobject("lock.png")
        lock1.scale(0.5)  # Increased size
        lock1.move_to([mid1[0], mid1[1], 0])

        # Lock for arrow2 (middle line) at same x-coordinate as others
        lock2 = ImageMobject("lock.png")
        lock2.scale(0.5)  # Increased size
        lock2.move_to([mid1[0], 0, 0])  # Same x as other locks, on middle line

        # Lock for arrow3 inflection point (new calculated position) - larger size
        lock3 = ImageMobject("lock.png")
        lock3.scale(0.5)  # Increased size
        lock3.move_to([mid3[0], mid3[1], 0])

        # Show locks just before lines finish
        self.play(
            FadeIn(lock1),
            FadeIn(lock2),
            FadeIn(lock3),
            run_time=0.5
        )
        self.wait(0.5)

        # Create final aggregated weight matrix with black background
        # Simple average for demonstration
        final_data = (weight1_data + weight2_data + weight3_data) / 3
        final_weight = create_weight_matrix(final_data, "", PURPLE)  # No label in matrix

        # Create standalone "Aggregated Weights" text
        aggregated_text = Text("Aggregated Weights", font_size=24)
        aggregated_text.next_to(final_weight, DOWN, buff=0.1)

        # Calculate final weight position past the end of the lines
        final_x = weight2.get_right()[0] + 2 * segment_length + 1.5  # Extended past line ends

        # Position final weight elements
        final_weight.move_to([2, 0, 0])
        final_weight.set_z_index(10)  # Put weights in front of everything
        aggregated_text.next_to(final_weight, DOWN, buff=0.1)

        # Animate the creation of final weight matrix
        self.play(
            FadeIn(final_weight),
            FadeIn(aggregated_text),
            run_time=2
        )
        self.wait(2)


        # Fade out everything except final weight (devices, weights, and locks)
        everything_else = Group(
            device1, device2, device3,
            weight1, weight2, weight3,
            arrow1, arrow2, arrow3,
            lock1, lock2, lock3,
            aggregated_text  # Include text to fade out
        )

        # Create final message text
        final_text = Text("Global Model Updated!", font_size=36, color=PURPLE)
        final_text.to_edge(DOWN)

        self.play(
            FadeOut(everything_else),
            final_weight.animate.move_to([0, 0, 0]).scale(1.5),
            FadeIn(final_text),
            run_time=2
        )
        self.wait(1)

        # Final message
        final_text = Text("Global Model Updated!", font_size=36, color=PURPLE)
        final_text.to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2)

        # Create the evergreen tree behind the weights
        tree_png_path = "evergreen_tree.png"
        final_tree = ImageMobject(tree_png_path)
        final_tree.scale(2.5)  # Larger tree size
        final_tree.move_to([0, 0, 0])
        final_tree.set_z_index(-1)  # Put tree behind weights

        # Fade in tree simultaneously as weights fade out
        self.play(
            FadeOut(final_weight),
            FadeIn(final_tree),
            run_time=1.5
        )
        self.wait(2)


if __name__ == "__main__":
    import subprocess
    import sys
    import os

    # Get the script directory and filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)

    # Flag for high quality output
    HIGH_QUALITY = True  # Set to True for full fps and resolution

    if HIGH_QUALITY:
        # High quality settings - MP4 at 60fps
        cmd = [
            "uv", "run", "manim",
            script_name,
            "WeightOverlayAnimation",
            "--format=mp4",  # MP4 instead of GIF
            "-qh",  # High quality
            "--disable_caching",
            "--fps", "60",  # Full fps
            "--media_dir", ".",  # Save to current directory (media)
            "--progress_bar", "display"
        ]
    else:
        # Low quality settings (default)
        cmd = [
            "uv", "run", "manim",
            script_name,
            "WeightOverlayAnimation",
            "--format=gif",
            "-ql",  # Low quality
            "--disable_caching",
            "--fps", "10",
            "--media_dir", ".",  # Save to current directory (media)
            "--progress_bar", "display"
        ]

    print(f"Running: {' '.join(cmd)}")

    try:
        # Change to script directory and run manim
        result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Animation GIF generated successfully!")
            print(result.stdout)

            # Find and display the output file path
            output_path = None
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'File ready at' in line and ('.gif' in line or '.mp4' in line):
                    output_path = line.split('File ready at')[1].strip()
                    print(f"Animation saved to: {output_path}")
                    break

            # Only speed up GIFs, not MP4s
            if output_path and '.gif' in output_path and os.path.exists(output_path):
                from PIL import Image

                print(f"Speeding up GIF by 10x using PIL...")

                try:
                    # Open the original GIF
                    gif = Image.open(output_path)

                    # Extract all frames
                    frames = []
                    durations = []

                    for frame_num in range(gif.n_frames):
                        gif.seek(frame_num)
                        frames.append(gif.copy())
                        # Get original duration and divide by 10 (speed up 10x)
                        duration = gif.info.get('duration', 100) // 10
                        durations.append(max(duration, 10))  # Minimum 10ms per frame

                    # Save the sped-up GIF
                    frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=0
                    )

                    print(f"✓ GIF sped up by 10x and saved to: {output_path}")

                except Exception as e:
                    print(f"✗ Error speeding up GIF with PIL: {e}")
            elif output_path and '.mp4' in output_path:
                print(f"✓ MP4 generated at full 60fps: {output_path}")
            else:
                print(f"✗ Output file not found")
        else:
            print("✗ Error generating animation:")
            print(result.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"✗ Failed to run manim: {e}")
        sys.exit(1)