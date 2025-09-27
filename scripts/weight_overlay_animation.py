#!/usr/bin/env python3
"""Manim animation showing 3 weight matrices being overlayed to create a final weight."""

from manim import *
import numpy as np


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

        # Create the three weight matrices
        weight1 = create_weight_matrix(weight1_data, "Client 1", BLUE)
        weight2 = create_weight_matrix(weight2_data, "Client 2", RED)
        weight3 = create_weight_matrix(weight3_data, "Client 3", GREEN)

        # Stack matrices on the left
        weight1.move_to([-5, 2.5, 0])
        weight2.move_to([-5, 0, 0])
        weight3.move_to([-5, -2.5, 0])

        # Animate the appearance of weight matrices
        self.play(
            FadeIn(weight1),
            FadeIn(weight2),
            FadeIn(weight3),
            run_time=2
        )
        self.wait(1)

        # Create connecting lines (without arrow heads)
        # Top line: goes right then down
        arrow1 = VGroup()
        start1 = weight1.get_right() + RIGHT * 0.1
        mid1 = [0.5, 2.5, 0]
        end1 = [2, 0.5, 0]
        line1_h = Line(start1, mid1, color=BLUE, stroke_width=3)
        line1_v = Line(mid1, end1, color=BLUE, stroke_width=3)
        arrow1.add(line1_h, line1_v)

        # Middle line: straight across
        arrow2 = Line(
            weight2.get_right() + RIGHT * 0.1,
            [2, 0, 0],
            color=RED,
            stroke_width=3
        )

        # Bottom line: goes right then up
        arrow3 = VGroup()
        start3 = weight3.get_right() + RIGHT * 0.1
        mid3 = [0.5, -2.5, 0]
        end3 = [2, -0.5, 0]
        line3_h = Line(start3, mid3, color=GREEN, stroke_width=3)
        line3_v = Line(mid3, end3, color=GREEN, stroke_width=3)
        arrow3.add(line3_h, line3_v)

        self.play(
            Create(arrow1),
            Create(arrow2),
            Create(arrow3),
            run_time=1.5
        )
        self.wait(0.5)

        # Create final aggregated weight matrix
        # Simple average for demonstration
        final_data = (weight1_data + weight2_data + weight3_data) / 3
        final_weight = create_weight_matrix(final_data, "Aggregated Weights", PURPLE)
        final_weight.move_to([2, 0, 0])

        # Animate the creation of final weight
        self.play(FadeIn(final_weight), run_time=2)
        self.wait(2)

        # Highlight the aggregation process
        highlight_group = VGroup(weight1, weight2, weight3, final_weight)
        self.play(
            highlight_group.animate.set_stroke(YELLOW, width=3),
            run_time=1
        )
        self.wait(1)

        # Fade out everything except final weight
        everything_else = VGroup(
            weight1, weight2, weight3,
            arrow1, arrow2, arrow3
        )

        self.play(
            FadeOut(everything_else),
            final_weight.animate.move_to([0, 0, 0]).scale(1.5),
            run_time=2
        )

        # Final message
        final_text = Text("Global Model Updated!", font_size=36, color=PURPLE)
        final_text.to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2)


if __name__ == "__main__":
    # To render as GIF, run:
    # manim weight_overlay_animation.py WeightOverlayAnimation --format=gif -r 480,270 --fps 15
    # or for higher quality:
    # manim weight_overlay_animation.py WeightOverlayAnimation --format=gif -r 854,480 --fps 30
    pass