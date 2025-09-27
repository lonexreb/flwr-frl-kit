#!/usr/bin/env python3
"""Manim animation showing 3 weight matrices being overlayed to create a final weight."""

from manim import *
import numpy as np
import os


class WeightOverlayAnimation(Scene):
    def perform_client_update(self, client_id, devices, weights, current_global_data, segment_length, mid_points, end_points, final_weight, create_weight_matrix, update_global=True):
        """Perform a client update animation with flash, weight change, and dot movement"""

        # Client configuration (moved down by 0.15)
        client_configs = {
            1: {"device": devices[0], "weight": weights[0], "color": BLUE, "label": "Client 1",
                "position": [-4.5, 2.35, 0], "line_y": 2.5},
            2: {"device": devices[1], "weight": weights[1], "color": RED, "label": "Client 2",
                "position": [-4.5, -0.15, 0], "line_y": 0},
            3: {"device": devices[2], "weight": weights[2], "color": GREEN, "label": "Client 3",
                "position": [-4.5, -2.65, 0], "line_y": -2.5}
        }

        config = client_configs[client_id]

        # Create new random weights for client after local training
        new_client_data = np.random.rand(4, 4) * 0.5

        # Update existing weight matrix squares with new values
        weight_matrix = config["weight"]
        squares = weight_matrix[:-1]  # All except the label (last element)

        # Create yellow dot at client to propagate update to global model
        update_dot = Dot(color=YELLOW, radius=0.1)
        update_dot.move_to(config["weight"].get_center())

        # Create path for update dot (client to global model)
        if client_id == 2:  # Middle client has straight line
            update_path = VMobject()
            update_path.set_points_as_corners([
                [config["weight"].get_center()[0], 0, 0],  # Start at client position
                [config["weight"].get_right()[0] + 2 * segment_length + 0.1, 0, 0],  # Go to line end
                final_weight.get_center()  # End at global model
            ])
        else:  # Top and bottom clients have angled lines
            mid_point = mid_points[0] if client_id == 1 else mid_points[2]
            end_point = end_points[0] if client_id == 1 else end_points[2]
            update_path = VMobject()
            update_path.set_points_as_corners([
                [config["weight"].get_center()[0], config["line_y"], 0],  # Start at client position
                mid_point,  # Go to inflection point
                end_point,  # Go to line end
                final_weight.get_center()  # End at global model
            ])

        # Create animations for updating individual squares
        square_animations = []
        square_index = 0
        for i in range(4):
            for j in range(4):
                value = new_client_data[i, j]
                square = squares[square_index]

                if value >= 0:
                    # Positive values: use the client's color scheme
                    intensity = min(abs(value) * 2, 1.0)  # Scale for [0, 0.5] range
                    new_color = config["color"]
                    new_opacity = intensity
                else:
                    # Negative values: use opposite hue of the respective color
                    intensity = min(abs(value) * 2, 1.0)  # Scale for [-0.5, 0] range

                    # Map to opposite hue colors
                    if config["color"] == BLUE:
                        new_color = ORANGE  # Opposite of blue
                    elif config["color"] == RED:
                        new_color = CYAN    # Opposite of red
                    elif config["color"] == GREEN:
                        new_color = PINK    # Opposite of green
                    else:
                        new_color = WHITE   # Fallback

                    new_opacity = intensity

                square_animations.append(square.animate.set_fill(new_color, opacity=new_opacity))
                square_index += 1

        # Update weights (no flashing, no dot if not updating global)
        if update_global:
            self.play(
                *square_animations,  # Update all squares individually
                FadeIn(update_dot),
                run_time=0.3
            )
        else:
            self.play(
                *square_animations,  # Update all squares individually
                run_time=0.3
            )

        # Only update global model if requested
        if update_global:
            # Prepare updated global model: (1/3 * new_client + 2/3 * current_global)
            updated_global_data = (1/3 * new_client_data + 2/3 * current_global_data)

            # Create animations for updating global model squares
            global_square_animations = []
            global_squares = final_weight[:-1]  # All except the label (last element)
            global_square_index = 0
            for i in range(4):
                for j in range(4):
                    value = updated_global_data[i, j]
                    square = global_squares[global_square_index]

                    if value >= 0:
                        # Positive values: use purple
                        intensity = min(abs(value) * 2, 1.0)  # Scale for [0, 0.5] range
                        new_color = PURPLE
                        new_opacity = intensity
                    else:
                        # Negative values: use opposite hue
                        intensity = min(abs(value) * 2, 1.0)  # Scale for [-0.5, 0] range
                        new_color = YELLOW  # Opposite of purple
                        new_opacity = intensity

                    global_square_animations.append(square.animate.set_fill(new_color, opacity=new_opacity))
                    global_square_index += 1

            # Client dot movement with immediate global model update when dot arrives
            self.play(
                MoveAlongPath(update_dot, update_path),  # Dot movement
                run_time=0.8  # Faster dot movement
            )
            # Update global model immediately as dot arrives
            self.play(
                *global_square_animations,  # Update all global model squares individually
                FadeOut(update_dot),
                run_time=0.1  # Much faster update
            )

            return updated_global_data
        else:
            # No global update, return current data unchanged
            return current_global_data

    def construct(self):
        # Set black background
        self.camera.background_color = BLACK

        # Create 3 weight matrices (represented as grids with colors) - range [0, 0.5]
        weight1_data = np.random.rand(4, 4) * 0.5  # Client 1 weights
        weight2_data = np.random.rand(4, 4) * 0.5  # Client 2 weights
        weight3_data = np.random.rand(4, 4) * 0.5  # Client 3 weights

        # Create visual representations of weight matrices
        def create_weight_matrix(data, label, color_scheme):
            # Create a grid of squares representing the matrix
            matrix_group = VGroup()

            for i in range(4):
                for j in range(4):
                    # Create square with color based on weight value (positive/negative)
                    value = data[i, j]
                    square = Square(side_length=0.4)

                    if value >= 0:
                        # Positive values: use the provided color scheme
                        intensity = min(abs(value) * 2, 1.0)  # Scale for [0, 0.5] range
                        square.set_fill(color_scheme, opacity=intensity)
                    else:
                        # Negative values: use opposite hue of the respective color
                        intensity = min(abs(value) * 2, 1.0)  # Scale for [-0.5, 0] range

                        # Map to opposite hue colors
                        if color_scheme == BLUE:
                            opposite_color = ORANGE  # Opposite of blue
                        elif color_scheme == RED:
                            opposite_color = CYAN    # Opposite of red
                        elif color_scheme == GREEN:
                            opposite_color = PINK    # Opposite of green
                        elif color_scheme == PURPLE:
                            opposite_color = YELLOW  # Opposite of purple
                        else:
                            opposite_color = WHITE   # Fallback

                        square.set_fill(opposite_color, opacity=intensity)

                    square.set_stroke(WHITE, width=1)
                    square.move_to([j * 0.5 - 0.75, -i * 0.5 + 0.75, 0])
                    matrix_group.add(square)

            # Add label
            label_text = Text(label, font_size=24)
            label_text.next_to(matrix_group, DOWN, buff=0.1)
            matrix_group.add(label_text)

            return matrix_group



        # Create device emojis for each client
        script_dir = os.path.dirname(os.path.abspath(__file__))

        device1 = ImageMobject("phone.png")
        device1.scale(1.0)
        device1.move_to([-6.5, 2.35, 0])

        device2 = ImageMobject("robot.png")
        device2.scale(1.0)
        device2.move_to([-6.5, -0.15, 0])

        device3 = ImageMobject("laptop.png")
        device3.scale(1.0)
        device3.move_to([-6.5, -2.65, 0])


        # Initialize client weights to zeros
        zero_data = np.zeros((4, 4))
        weight1 = create_weight_matrix(zero_data, "Client 1", BLUE)
        weight2 = create_weight_matrix(zero_data, "Client 2", RED)
        weight3 = create_weight_matrix(zero_data, "Client 3", GREEN)

        # Stack matrices to match device positions (moved down by 0.15)
        weight1.move_to([-4.5, 2.35, 0])
        weight2.move_to([-4.5, -0.15, 0])
        weight3.move_to([-4.5, -2.65, 0])

        # Create final aggregated weight matrix first - range [0, 0.5]
        final_data = (weight1_data + weight2_data + weight3_data) / 3
        final_weight = create_weight_matrix(final_data, "", PURPLE)
        aggregated_text = Text("Global Model", font_size=24)

        # Position final weight elements
        final_weight.move_to([1, 0, 0])
        aggregated_text.next_to(final_weight, DOWN, buff=0.1)

        # Create phase text in top right corner
        phase_text = Text("Server Initialization", font_size=28, color=WHITE)
        phase_text.move_to([4, 3, 0])  # Top right corner

        # Show server/global model first
        self.play(
            FadeIn(final_weight),
            FadeIn(aggregated_text),
            FadeIn(phase_text),
            run_time=2
        )

        self.wait(1)

        # Update phase text to Client Initialization
        client_phase_text = Text("Client Initialization", font_size=28, color=WHITE)
        client_phase_text.move_to([4, 3, 0])

        # Then show clients and their initial weights
        self.play(
            FadeIn(device1),
            FadeIn(device2),
            FadeIn(device3),
            FadeIn(weight1),
            FadeIn(weight2),
            FadeIn(weight3),
            Transform(phase_text, client_phase_text),
            run_time=2
        )

        self.wait(1)

        # Create connecting lines with 45-degree curved elbows and equal segment lengths
        # Use original positions for line start points (before shifting)
        segment_length = 1.5

        # Top line: goes right then down at 45 degrees (from original position)
        arrow1 = VGroup()
        start1 = [-4.5 + 1.0, 2.5, 0] + RIGHT * 0.1  # Original position + matrix width
        mid1 = [start1[0] + segment_length, 2.5, 0]  # Keep original Y coordinate
        end1 = [mid1[0] + segment_length, 0, 0]  # End at original middle line Y
        line1_h = Line(start1, mid1, color=BLUE, stroke_width=3)
        line1_v = Line(mid1, end1, color=BLUE, stroke_width=3)
        arrow1.add(line1_h, line1_v)

        # Middle line: goes right (from original position)
        arrow2 = Line(
            [-4.5 + 1.0, 0, 0] + RIGHT * 0.1,  # Original position + matrix width
            [-4.5 + 1.0 + 2 * segment_length + 0.1, 0, 0],
            color=RED,
            stroke_width=3
        )

        # Bottom line: goes right then up at 45 degrees (from original position)
        arrow3 = VGroup()
        start3 = [-4.5 + 1.0, -2.5, 0] + RIGHT * 0.1  # Original position + matrix width
        mid3 = [start3[0] + segment_length, -2.5, 0]  # Keep original Y coordinate
        end3 = [mid3[0] + segment_length, 0, 0]  # End at original middle line Y
        line3_h = Line(start3, mid3, color=GREEN, stroke_width=3)
        line3_v = Line(mid3, end3, color=GREEN, stroke_width=3)
        arrow3.add(line3_h, line3_v)

        # Store key points for client update method
        devices = [device1, device2, device3]
        weights = [weight1, weight2, weight3]
        mid_points = [mid1, None, mid3]  # Middle client doesn't use mid point
        end_points = [end1, None, end3]  # Middle client doesn't use end point

        self.play(
            Create(arrow1),
            Create(arrow2),
            Create(arrow3),
            run_time=1.5
        )

        # Create lock emojis at inflection points and on middle line
        script_dir = os.path.dirname(os.path.abspath(__file__))

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

        # Create dots that will travel from global model to clients
        dot1 = Dot(color=BLUE, radius=0.1)
        dot2 = Dot(color=RED, radius=0.1)
        dot3 = Dot(color=GREEN, radius=0.1)

        # Start dots at the line endpoints (near global model)
        dot1.move_to(end1)  # End of arrow1 (top line)
        dot2.move_to([weight2.get_right()[0] + 2 * segment_length + 0.1, 0, 0])  # End of arrow2 (middle line)
        dot3.move_to(end3)  # End of arrow3 (bottom line)

        # Show dots at line endpoints
        self.play(
            FadeIn(dot1),
            FadeIn(dot2),
            FadeIn(dot3),
            run_time=0.5
        )

        # Create paths for dots to follow their respective lines
        from manim import VMobject

        # Create path for top dot (two segments)
        path1 = VMobject()
        path1.set_points_as_corners([
            end1,  # Start at line end
            mid1,  # Go to inflection point
            [weight1.get_center()[0], 2.5, 0]  # End at client position
        ])

        # Create path for middle dot (straight line)
        path2 = VMobject()
        path2.set_points_as_corners([
            [weight2.get_right()[0] + 2 * segment_length + 0.1, 0, 0],  # Start at line end
            [weight2.get_center()[0], 0, 0]  # End at client position
        ])

        # Create path for bottom dot (two segments)
        path3 = VMobject()
        path3.set_points_as_corners([
            end3,  # Start at line end
            mid3,  # Go to inflection point
            [weight3.get_center()[0], -2.5, 0]  # End at client position
        ])

        # Animate all dots following their paths for the same duration
        self.play(
            MoveAlongPath(dot1, path1),
            MoveAlongPath(dot2, path2),
            MoveAlongPath(dot3, path3),
            run_time=2
        )

        # Create updated weight matrices with global model data - range [0, 0.5]
        global_model_data = (weight1_data + weight2_data + weight3_data) / 3
        updated_weight1 = create_weight_matrix(global_model_data, "Client 1", BLUE)
        updated_weight2 = create_weight_matrix(global_model_data, "Client 2", RED)
        updated_weight3 = create_weight_matrix(global_model_data, "Client 3", GREEN)

        # Position updated weights to match device positions (moved down by 0.15)
        updated_weight1.move_to([-4.5, 2.35, 0])
        updated_weight2.move_to([-4.5, -0.15, 0])
        updated_weight3.move_to([-4.5, -2.65, 0])

        # Update phase text to Initial Weight Population
        weight_phase_text = Text("Initial Weight Population", font_size=28, color=WHITE)
        weight_phase_text.move_to([4, 3, 0])

        # Replace zero weights with updated weights and fade out dots
        self.play(
            Transform(weight1, updated_weight1),
            Transform(weight2, updated_weight2),
            Transform(weight3, updated_weight3),
            Transform(phase_text, weight_phase_text),
            FadeOut(dot1),
            FadeOut(dot2),
            FadeOut(dot3),
            run_time=1.5
        )
        self.wait(1)

        # Track current global model data for updates
        current_global_data = final_data

        # Update phase text to Federated Training
        training_phase_text = Text("Federated Training", font_size=28, color=WHITE)
        training_phase_text.move_to([4, 3, 0])

        self.play(
            Transform(phase_text, training_phase_text),
            run_time=1
        )

        # Distributed client updates with fast timing
        self.wait(0.1)  # Brief initial wait

        # Client 1 first update
        update_global = np.random.choice([True, False])  # 50% chance
        current_global_data = self.perform_client_update(1, devices, weights, current_global_data, segment_length, mid_points, end_points, final_weight, create_weight_matrix, update_global)

        self.wait(0.2)  # Short wait before next update

        # Client 3 update
        update_global = np.random.choice([True, False])  # 50% chance
        current_global_data = self.perform_client_update(3, devices, weights, current_global_data, segment_length, mid_points, end_points, final_weight, create_weight_matrix, update_global)

        self.wait(0.1)  # Minimal wait

        # Client 2 first update
        update_global = np.random.choice([True, False])  # 50% chance
        current_global_data = self.perform_client_update(2, devices, weights, current_global_data, segment_length, mid_points, end_points, final_weight, create_weight_matrix, update_global)

        self.wait(0.15)  # Short wait

        # Client 1 second update
        update_global = np.random.choice([True, False])  # 50% chance
        current_global_data = self.perform_client_update(1, devices, weights, current_global_data, segment_length, mid_points, end_points, final_weight, create_weight_matrix, update_global)

        self.wait(0.1)  # Minimal wait

        # Client 3 second update
        update_global = np.random.choice([True, False])  # 50% chance
        current_global_data = self.perform_client_update(3, devices, weights, current_global_data, segment_length, mid_points, end_points, final_weight, create_weight_matrix, update_global)

        self.wait(0.2)  # Short wait

        # Client 2 second update
        update_global = np.random.choice([True, False])  # 50% chance
        current_global_data = self.perform_client_update(2, devices, weights, current_global_data, segment_length, mid_points, end_points, final_weight, create_weight_matrix, update_global)

        # Additional rapid updates to show continuous learning
        for _ in range(4):
            self.wait(0.05)  # Very short waits
            client_id = np.random.choice([1, 2, 3])  # Random client
            update_global = np.random.choice([True, False])  # 50% chance
            current_global_data = self.perform_client_update(client_id, devices, weights, current_global_data, segment_length, mid_points, end_points, final_weight, create_weight_matrix, update_global)

        self.wait(1)

        # Update phase text to Training Complete
        complete_phase_text = Text("Training Complete", font_size=28, color=WHITE)
        complete_phase_text.move_to([4, 3, 0])

        self.play(
            Transform(phase_text, complete_phase_text),
            run_time=1
        )

        self.wait(0.5)

        # Clear everything except global model
        self.play(
            FadeOut(device1),
            FadeOut(device2),
            FadeOut(device3),
            FadeOut(weight1),
            FadeOut(weight2),
            FadeOut(weight3),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(arrow3),
            FadeOut(lock1),
            FadeOut(lock2),
            FadeOut(lock3),
            run_time=2
        )

        self.wait(0.5)

        # Create RL environment depictions

        # 1. Reverse Pendulum (top-left)
        pendulum_group = VGroup()
        # Base
        base = Rectangle(width=0.8, height=0.2, color=WHITE, fill_opacity=0.8)
        # Pole
        pole = Line(start=[0, 0.1, 0], end=[0, 1.2, 0], color=GRAY, stroke_width=8)
        # Cart
        cart = Rectangle(width=0.4, height=0.3, color=LIGHT_GRAY, fill_opacity=1)
        cart.move_to([0, 0.25, 0])
        # Mass at top
        mass = Circle(radius=0.15, color=DARK_GRAY, fill_opacity=1)
        mass.move_to([0, 1.2, 0])
        pendulum_group.add(base, pole, cart, mass)
        pendulum_group.move_to([-4.5, 2.5, 0])

        # 2. Math equation (top-right)
        math_group = VGroup()
        equation = MathTex("x + 3 = 4", font_size=32, color=WHITE)
        arrow = MathTex("\\rightarrow", font_size=32, color=LIGHT_GRAY)
        question = MathTex("x = ?", font_size=32, color=GRAY)
        math_group.add(equation, arrow, question)
        math_group.arrange(RIGHT, buff=0.3)
        math_group.move_to([4.5, 2.5, 0])

        # 3. Robot (bottom-left)
        robot_group = VGroup()
        # Robot body
        robot_body = Rectangle(width=0.6, height=0.8, color=LIGHT_GRAY, fill_opacity=1)
        # Robot head
        robot_head = Circle(radius=0.25, color=LIGHT_GRAY, fill_opacity=1)
        robot_head.move_to([0, 0.6, 0])
        # Eyes
        left_eye = Circle(radius=0.05, color=DARK_GRAY, fill_opacity=1)
        right_eye = Circle(radius=0.05, color=DARK_GRAY, fill_opacity=1)
        left_eye.move_to([-0.1, 0.65, 0])
        right_eye.move_to([0.1, 0.65, 0])
        # Arms
        left_arm = Rectangle(width=0.15, height=0.4, color=GRAY, fill_opacity=1)
        right_arm = Rectangle(width=0.15, height=0.4, color=GRAY, fill_opacity=1)
        left_arm.move_to([-0.45, 0.2, 0])
        right_arm.move_to([0.45, 0.2, 0])
        # Legs
        left_leg = Rectangle(width=0.2, height=0.5, color=GRAY, fill_opacity=1)
        right_leg = Rectangle(width=0.2, height=0.5, color=GRAY, fill_opacity=1)
        left_leg.move_to([-0.15, -0.65, 0])
        right_leg.move_to([0.15, -0.65, 0])

        robot_group.add(robot_body, robot_head, left_eye, right_eye, left_arm, right_arm, left_leg, right_leg)
        robot_group.move_to([-4.5, -2.5, 0])

        # 4. Agent code (bottom-right)
        agent_group = VGroup()
        code_line1 = Text("def policy(state):", font_size=20, color=WHITE)
        code_line2 = Text("    action = model.predict(state)", font_size=20, color=LIGHT_GRAY)
        code_line3 = Text("    return action", font_size=20, color=GRAY)
        agent_group.add(code_line1, code_line2, code_line3)
        agent_group.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        agent_group.move_to([4.5, -2.5, 0])

        # Create bottom text about training policies
        bottom_text = Text("Train policies across devices; strategies; data sources", font_size=24, color=WHITE)
        bottom_text.move_to([0, -3.7, 0])

        # Scale up and center the global model, fade out training complete text and global model text
        self.play(
            FadeOut(phase_text),  # Remove "Training Complete" text
            FadeOut(aggregated_text),  # Fade out "Global Model" text
            final_weight.animate.scale(2.0).move_to([0, 0, 0]),
            FadeIn(bottom_text),  # Add new bottom text
            run_time=2
        )

        self.wait(0.5)

        # Show environments sequentially
        self.play(FadeIn(pendulum_group), run_time=0.8)
        self.wait(0.3)

        self.play(FadeIn(math_group), run_time=0.8)
        self.wait(0.3)

        self.play(FadeIn(robot_group), run_time=0.8)
        self.wait(0.3)

        self.play(FadeIn(agent_group), run_time=0.8)

        self.wait(3)


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