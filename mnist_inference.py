# ================================================================
# GENREG MNIST Inference Tool
# ================================================================
# Draw digits and test MNIST genome predictions in real-time.
#
# Features:
# - Select a best genome from best_genomes/ folder
# - Draw digits with mouse in a canvas
# - Real-time inference with timing metrics
# - Clear button to erase canvas
#
# Usage:
#   python mnist_inference.py
# ================================================================

import os
import sys
import pickle
import time
from collections import deque

import pygame
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genreg_controller import GENREGController


# ================================================================
# GENOME LOADING
# ================================================================
def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def list_mnist_genomes():
    """List all MNIST genomes in best_genomes folder."""
    genomes = []
    genome_dir = "best_genomes"

    if not os.path.exists(genome_dir):
        return []

    for filename in os.listdir(genome_dir):
        if filename.endswith('.pkl') and 'mnist' in filename.lower():
            filepath = os.path.join(genome_dir, filename)
            file_size = os.path.getsize(filepath)

            # Extract generation from filename
            gen_num = "?"
            try:
                if "_gen" in filename:
                    gen_part = filename.split("_gen")[1]
                    gen_num = gen_part.split("_")[0]
            except:
                pass

            genomes.append({
                'path': filepath,
                'filename': filename,
                'size': file_size,
                'generation': gen_num
            })

    # Sort by filename (newest first typically)
    genomes.sort(key=lambda x: x['filename'], reverse=True)
    return genomes


def select_genome():
    """Interactive genome selection."""
    genomes = list_mnist_genomes()

    if not genomes:
        print("No MNIST genomes found in best_genomes/")
        print("Extract one first with: python extract_best_genome.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SELECT MNIST GENOME")
    print("=" * 60)
    print("\nAvailable MNIST genomes:\n")

    for i, g in enumerate(genomes[:10]):
        size_str = format_size(g['size'])
        print(f"  [{i+1}] {g['filename']}")
        print(f"       Gen {g['generation']}, {size_str}")

    if len(genomes) == 1:
        print(f"\n  [Enter] Use {genomes[0]['filename']}")
    else:
        print(f"\n  [Enter] Use most recent")

    choice = input("\nYour choice: ").strip()

    if choice == "":
        return genomes[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(genomes):
        return genomes[int(choice) - 1]
    else:
        print("Invalid choice, using most recent...")
        return genomes[0]


def load_genome(genome_info):
    """Load a genome from pickle file."""
    print(f"\nLoading: {genome_info['filename']}")

    with open(genome_info['path'], 'rb') as f:
        data = pickle.load(f)

    # Extract controller data
    ctrl_data = data['controller']
    input_size = ctrl_data['input_size']
    hidden_size = ctrl_data['hidden_size']
    output_size = ctrl_data['output_size']

    print(f"  Network: {input_size} -> {hidden_size} -> {output_size}")

    # Verify it's a MNIST model
    if input_size != 784:
        print(f"  WARNING: Expected 784 input (MNIST), got {input_size}")
        print("  This may not be a MNIST model!")

    if output_size != 10:
        print(f"  WARNING: Expected 10 output (digits), got {output_size}")

    # Create controller and load weights
    controller = GENREGController(input_size, hidden_size, output_size)

    # Load weights
    if hasattr(controller, '_use_torch') and controller._use_torch:
        import torch
        controller.w1 = torch.tensor(ctrl_data['w1'], dtype=torch.float32, device=controller.device)
        controller.b1 = torch.tensor(ctrl_data['b1'], dtype=torch.float32, device=controller.device)
        controller.w2 = torch.tensor(ctrl_data['w2'], dtype=torch.float32, device=controller.device)
        controller.b2 = torch.tensor(ctrl_data['b2'], dtype=torch.float32, device=controller.device)
    else:
        controller.w1 = ctrl_data['w1']
        controller.b1 = ctrl_data['b1']
        controller.w2 = ctrl_data['w2']
        controller.b2 = ctrl_data['b2']

    # Get metadata
    trust = data.get('genome', {}).get('trust', 0)
    print(f"  Trust: {trust:.2f}")

    return controller, data


# ================================================================
# DRAWING CANVAS
# ================================================================
class DrawingCanvas:
    """28x28 drawing canvas that matches MNIST format exactly."""

    # MNIST native size - must match training!
    MNIST_SIZE = 28

    def __init__(self, display_size=280):
        self.display_size = display_size

        # High-res canvas for smooth drawing
        self.draw_surface = pygame.Surface((display_size, display_size))
        self.draw_surface.fill((0, 0, 0))

        # 28x28 surface for model input (what the model actually sees)
        self.mnist_surface = pygame.Surface((self.MNIST_SIZE, self.MNIST_SIZE))
        self.mnist_surface.fill((0, 0, 0))

        # Cached observation
        self.cached_obs = None
        self.obs_dirty = True

        # Brush settings (scaled for display size)
        self.brush_size = 20
        self.drawing = False
        self.last_pos = None

    def start_draw(self, pos):
        """Start drawing."""
        self.drawing = True
        self.last_pos = pos
        self._draw_circle(pos)
        self.obs_dirty = True

    def continue_draw(self, pos):
        """Continue drawing line."""
        if self.drawing and self.last_pos:
            self._draw_line(self.last_pos, pos)
            self.last_pos = pos
            self.obs_dirty = True

    def stop_draw(self):
        """Stop drawing."""
        self.drawing = False
        self.last_pos = None

    def _draw_circle(self, pos):
        """Draw a circle at position."""
        pygame.draw.circle(self.draw_surface, (255, 255, 255), pos, self.brush_size)

    def _draw_line(self, start, end):
        """Draw a line between two points."""
        pygame.draw.line(self.draw_surface, (255, 255, 255), start, end, self.brush_size * 2)
        pygame.draw.circle(self.draw_surface, (255, 255, 255), end, self.brush_size)

    def clear(self):
        """Clear the canvas."""
        self.draw_surface.fill((0, 0, 0))
        self.mnist_surface.fill((0, 0, 0))
        self.cached_obs = None
        self.obs_dirty = True

    def _update_mnist_surface(self):
        """Downscale drawing to 28x28 MNIST format."""
        # Use smoothscale for anti-aliased downsampling
        pygame.transform.smoothscale(
            self.draw_surface,
            (self.MNIST_SIZE, self.MNIST_SIZE),
            self.mnist_surface
        )

    def get_observation(self):
        """
        Get 28x28 normalized observation for model.

        Returns exactly 784 floats (28*28) in row-major order,
        matching MNIST training format.
        """
        if self.obs_dirty:
            # Downscale to 28x28
            self._update_mnist_surface()

            # Get pixel data from 28x28 surface
            # pygame.surfarray returns (width, height, channels)
            pixels = pygame.surfarray.array3d(self.mnist_surface)

            # Take grayscale (red channel, since we draw white on black)
            # pixels shape: (28, 28, 3) but pygame is (width, height)
            grayscale = pixels[:, :, 0].astype(np.float32)

            # Transpose from (width, height) to (height, width) for row-major order
            # This matches how MNIST data is stored: row by row from top to bottom
            grayscale = grayscale.T

            # Normalize to [0, 1]
            grayscale = grayscale / 255.0

            # Flatten in row-major order (C order) - 784 values
            self.cached_obs = grayscale.flatten().tolist()
            self.obs_dirty = False

        return self.cached_obs

    def get_surface(self):
        """Get the drawing surface for display."""
        return self.draw_surface

    def get_mnist_surface(self):
        """Get the 28x28 surface (what model sees) for preview."""
        if self.obs_dirty:
            self._update_mnist_surface()
        return self.mnist_surface


# ================================================================
# INFERENCE UI
# ================================================================
class MNISTInferenceUI:
    """Pygame UI for MNIST digit inference."""

    def __init__(self, controller, genome_info):
        self.controller = controller
        self.genome_info = genome_info

        # Window dimensions
        self.canvas_size = 280
        self.panel_width = 300
        self.preview_size = 56  # 28x28 scaled up 2x for visibility
        self.width = self.canvas_size + self.panel_width
        self.height = 450  # Taller to fit preview

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("GENREG MNIST Inference")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 96)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Drawing canvas
        self.canvas = DrawingCanvas(self.canvas_size)
        self.canvas_rect = pygame.Rect(0, 0, self.canvas_size, self.canvas_size)

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (100, 100, 100)
        self.DARK_GRAY = (50, 50, 50)
        self.GREEN = (50, 200, 50)
        self.RED = (200, 50, 50)
        self.BLUE = (50, 100, 200)
        self.YELLOW = (200, 200, 50)
        self.CYAN = (50, 200, 200)

        # Prediction state
        self.prediction = None
        self.confidence = 0.0
        self.all_probs = [0.0] * 10

        # Timing metrics
        self.inference_times = deque(maxlen=100)
        self.last_inference_time = 0.0
        self.total_inferences = 0

        # UI elements
        button_width = (self.panel_width - 50) // 2
        self.clear_button = pygame.Rect(
            self.canvas_size + 20, self.height - 70,
            button_width, 40
        )
        self.pause_button = pygame.Rect(
            self.canvas_size + 30 + button_width, self.height - 70,
            button_width, 40
        )
        self.benchmark_button = pygame.Rect(
            self.canvas_size + 20, self.height - 120,
            self.panel_width - 40, 35
        )

        # Auto-inference
        self.auto_infer = True
        self.paused = False
        self.last_draw_time = 0
        self.infer_delay = 0.1  # Seconds after drawing stops

        # Benchmark state
        self.benchmark_results = None
        self.benchmark_iterations = 1000

    def run_inference(self):
        """Run inference on current canvas."""
        # Get observation
        obs = self.canvas.get_observation()

        # Time the inference
        start_time = time.perf_counter()

        # Get prediction
        digit_probs = self.controller.forward_visual(obs)

        end_time = time.perf_counter()

        # Record timing
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        self.inference_times.append(inference_time)
        self.last_inference_time = inference_time
        self.total_inferences += 1

        # Get prediction (top digit)
        self.all_probs = digit_probs[:10]
        self.prediction = self.all_probs.index(max(self.all_probs))
        self.confidence = self.all_probs[self.prediction]

    def run_benchmark(self):
        """Run benchmark: multiple inferences on same input for accurate timing."""
        # Get observation once
        obs = self.canvas.get_observation()

        # Warmup runs
        for _ in range(10):
            self.controller.forward_visual(obs)

        # Benchmark runs
        times = []
        for _ in range(self.benchmark_iterations):
            start_time = time.perf_counter()
            digit_probs = self.controller.forward_visual(obs)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        # Calculate statistics
        times.sort()
        self.benchmark_results = {
            'iterations': self.benchmark_iterations,
            'min': min(times),
            'max': max(times),
            'mean': sum(times) / len(times),
            'median': times[len(times) // 2],
            'p95': times[int(len(times) * 0.95)],
            'p99': times[int(len(times) * 0.99)],
        }

        # Also update prediction
        self.all_probs = digit_probs[:10]
        self.prediction = self.all_probs.index(max(self.all_probs))
        self.confidence = self.all_probs[self.prediction]

        print(f"\n[BENCHMARK] {self.benchmark_iterations} iterations:")
        print(f"  Min:    {self.benchmark_results['min']:.4f} ms")
        print(f"  Max:    {self.benchmark_results['max']:.4f} ms")
        print(f"  Mean:   {self.benchmark_results['mean']:.4f} ms")
        print(f"  Median: {self.benchmark_results['median']:.4f} ms")
        print(f"  P95:    {self.benchmark_results['p95']:.4f} ms")
        print(f"  P99:    {self.benchmark_results['p99']:.4f} ms")
        print(f"  Throughput: {1000 / self.benchmark_results['mean']:.0f} inf/sec")

    def draw(self):
        """Draw the UI."""
        # Background
        self.screen.fill(self.DARK_GRAY)

        # Drawing canvas area
        pygame.draw.rect(self.screen, self.BLACK, self.canvas_rect)
        self.screen.blit(self.canvas.get_surface(), (0, 0))

        # Canvas border
        pygame.draw.rect(self.screen, self.GRAY, self.canvas_rect, 2)

        # Canvas label
        label = self.font_small.render("Draw a digit (0-9)", True, self.GRAY)
        self.screen.blit(label, (self.canvas_size // 2 - label.get_width() // 2, self.canvas_size + 5))

        # 28x28 Preview (what model sees)
        preview_label = self.font_small.render("Model sees (28x28):", True, self.GRAY)
        self.screen.blit(preview_label, (10, self.canvas_size + 25))

        # Get and scale the 28x28 surface
        mnist_surface = self.canvas.get_mnist_surface()
        preview_scaled = pygame.transform.scale(mnist_surface, (self.preview_size, self.preview_size))

        # Draw preview with border
        preview_x = 10
        preview_y = self.canvas_size + 45
        pygame.draw.rect(self.screen, self.GRAY, (preview_x - 1, preview_y - 1, self.preview_size + 2, self.preview_size + 2), 1)
        self.screen.blit(preview_scaled, (preview_x, preview_y))

        # Show observation size
        obs_info = self.font_small.render("784 pixels", True, self.GRAY)
        self.screen.blit(obs_info, (preview_x + self.preview_size + 10, preview_y + 20))

        # ========================================
        # SIDE PANEL
        # ========================================
        panel_x = self.canvas_size + 20
        y = 20

        # Title
        title = self.font_medium.render("MNIST Inference", True, self.WHITE)
        self.screen.blit(title, (panel_x, y))
        y += 40

        # Prediction
        if self.prediction is not None:
            # Large prediction display
            pred_text = self.font_large.render(str(self.prediction), True, self.CYAN)
            self.screen.blit(pred_text, (panel_x + 30, y))

            # Confidence bar
            conf_x = panel_x + 100
            conf_width = 150
            conf_height = 30

            # Background bar
            pygame.draw.rect(self.screen, self.GRAY,
                             (conf_x, y + 20, conf_width, conf_height))
            # Filled bar
            fill_width = int(conf_width * self.confidence)
            color = self.GREEN if self.confidence > 0.7 else self.YELLOW if self.confidence > 0.4 else self.RED
            pygame.draw.rect(self.screen, color,
                             (conf_x, y + 20, fill_width, conf_height))
            # Border
            pygame.draw.rect(self.screen, self.WHITE,
                             (conf_x, y + 20, conf_width, conf_height), 1)

            # Confidence text
            conf_text = self.font_small.render(f"{self.confidence * 100:.1f}%", True, self.WHITE)
            self.screen.blit(conf_text, (conf_x + conf_width + 10, y + 25))

            y += 80

            # All probabilities
            prob_label = self.font_small.render("All Probabilities:", True, self.GRAY)
            self.screen.blit(prob_label, (panel_x, y))
            y += 25

            # Draw probability bars for each digit
            bar_height = 12
            bar_max_width = 120
            for digit in range(10):
                prob = self.all_probs[digit]

                # Digit label
                digit_text = self.font_small.render(f"{digit}:", True,
                                                     self.CYAN if digit == self.prediction else self.WHITE)
                self.screen.blit(digit_text, (panel_x, y))

                # Bar
                bar_width = int(bar_max_width * prob)
                bar_color = self.CYAN if digit == self.prediction else self.GRAY
                pygame.draw.rect(self.screen, bar_color,
                                 (panel_x + 25, y + 2, bar_width, bar_height))
                pygame.draw.rect(self.screen, self.GRAY,
                                 (panel_x + 25, y + 2, bar_max_width, bar_height), 1)

                # Probability text
                prob_text = self.font_small.render(f"{prob * 100:.1f}%", True, self.WHITE)
                self.screen.blit(prob_text, (panel_x + 25 + bar_max_width + 5, y))

                y += bar_height + 4

        else:
            hint = self.font_small.render("Draw to predict", True, self.GRAY)
            self.screen.blit(hint, (panel_x, y + 40))
            y += 200

        # ========================================
        # TIMING METRICS
        # ========================================
        y = self.height - 220

        # Show paused status
        if self.paused:
            status_text = self.font_small.render("PAUSED - Draw then benchmark", True, self.YELLOW)
            self.screen.blit(status_text, (panel_x, y))
            y += 20
        else:
            status_text = self.font_small.render("AUTO-INFERENCE ON", True, self.GREEN)
            self.screen.blit(status_text, (panel_x, y))
            y += 20

        metrics_label = self.font_small.render("Inference Metrics:", True, self.GRAY)
        self.screen.blit(metrics_label, (panel_x, y))
        y += 20

        # Show benchmark results if available
        if self.benchmark_results:
            br = self.benchmark_results
            bench_text = self.font_small.render(f"Benchmark ({br['iterations']} runs):", True, self.CYAN)
            self.screen.blit(bench_text, (panel_x, y))
            y += 18

            mean_text = self.font_small.render(f"  Mean: {br['mean']:.3f} ms", True, self.WHITE)
            self.screen.blit(mean_text, (panel_x, y))
            y += 16

            median_text = self.font_small.render(f"  Median: {br['median']:.3f} ms", True, self.WHITE)
            self.screen.blit(median_text, (panel_x, y))
            y += 16

            throughput = 1000 / br['mean'] if br['mean'] > 0 else 0
            tp_text = self.font_small.render(f"  Throughput: {throughput:.0f}/sec", True, self.GREEN)
            self.screen.blit(tp_text, (panel_x, y))
            y += 20

        # Last inference time (when not paused)
        elif self.last_inference_time > 0:
            time_text = self.font_small.render(f"Last: {self.last_inference_time:.3f} ms", True, self.WHITE)
            self.screen.blit(time_text, (panel_x, y))
            y += 18

            # Average inference time
            if self.inference_times:
                avg_time = sum(self.inference_times) / len(self.inference_times)
                avg_text = self.font_small.render(f"Avg: {avg_time:.3f} ms", True, self.WHITE)
                self.screen.blit(avg_text, (panel_x, y))

                # FPS equivalent
                fps = 1000 / avg_time if avg_time > 0 else 0
                fps_text = self.font_small.render(f"({fps:.0f}/sec)", True, self.GRAY)
                self.screen.blit(fps_text, (panel_x + 120, y))
            y += 18

            # Total inferences
            total_text = self.font_small.render(f"Total: {self.total_inferences}", True, self.WHITE)
            self.screen.blit(total_text, (panel_x, y))

        # ========================================
        # BENCHMARK BUTTON (only when paused)
        # ========================================
        if self.paused:
            pygame.draw.rect(self.screen, self.CYAN, self.benchmark_button)
            pygame.draw.rect(self.screen, self.WHITE, self.benchmark_button, 2)
            bench_text = self.font_small.render("BENCHMARK (1000x)", True, self.BLACK)
            text_rect = bench_text.get_rect(center=self.benchmark_button.center)
            self.screen.blit(bench_text, text_rect)
        else:
            # Show dimmed button when not paused
            pygame.draw.rect(self.screen, self.DARK_GRAY, self.benchmark_button)
            pygame.draw.rect(self.screen, self.GRAY, self.benchmark_button, 1)
            bench_text = self.font_small.render("Pause to benchmark", True, self.GRAY)
            text_rect = bench_text.get_rect(center=self.benchmark_button.center)
            self.screen.blit(bench_text, text_rect)

        # ========================================
        # CLEAR AND PAUSE BUTTONS
        # ========================================
        # Clear button
        pygame.draw.rect(self.screen, self.RED, self.clear_button)
        pygame.draw.rect(self.screen, self.WHITE, self.clear_button, 2)
        clear_text = self.font_small.render("CLEAR", True, self.WHITE)
        text_rect = clear_text.get_rect(center=self.clear_button.center)
        self.screen.blit(clear_text, text_rect)

        # Pause/Resume button
        if self.paused:
            pygame.draw.rect(self.screen, self.GREEN, self.pause_button)
            pause_text = self.font_small.render("RESUME", True, self.BLACK)
        else:
            pygame.draw.rect(self.screen, self.YELLOW, self.pause_button)
            pause_text = self.font_small.render("PAUSE", True, self.BLACK)
        pygame.draw.rect(self.screen, self.WHITE, self.pause_button, 2)
        text_rect = pause_text.get_rect(center=self.pause_button.center)
        self.screen.blit(pause_text, text_rect)

        # ========================================
        # INSTRUCTIONS
        # ========================================
        instr = self.font_small.render("ESC quit | C clear | P/Space pause | B benchmark", True, self.GRAY)
        self.screen.blit(instr, (10, self.height - 20))

        pygame.display.flip()

    def run(self):
        """Main UI loop."""
        running = True

        print("\n[INFERENCE] Ready! Draw digits in the canvas.")
        print("[INFERENCE] Controls:")
        print("  ESC     - Quit")
        print("  C       - Clear canvas")
        print("  P/Space - Pause/Resume auto-inference")
        print("  B       - Run benchmark (when paused)\n")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        self.canvas.clear()
                        self.prediction = None
                        self.confidence = 0.0
                        self.all_probs = [0.0] * 10
                        self.benchmark_results = None
                    elif event.key == pygame.K_p or event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        if not self.paused:
                            self.benchmark_results = None
                        print(f"[{'PAUSED' if self.paused else 'RESUMED'}]")
                    elif event.key == pygame.K_b and self.paused:
                        print("[BENCHMARK] Running 1000 iterations...")
                        self.run_benchmark()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        pos = pygame.mouse.get_pos()

                        # Check clear button
                        if self.clear_button.collidepoint(pos):
                            self.canvas.clear()
                            self.prediction = None
                            self.confidence = 0.0
                            self.all_probs = [0.0] * 10
                            self.benchmark_results = None

                        # Check pause button
                        elif self.pause_button.collidepoint(pos):
                            self.paused = not self.paused
                            if not self.paused:
                                # Resuming - clear benchmark results
                                self.benchmark_results = None
                            print(f"[{'PAUSED' if self.paused else 'RESUMED'}]")

                        # Check benchmark button (only when paused)
                        elif self.benchmark_button.collidepoint(pos) and self.paused:
                            print("[BENCHMARK] Running 1000 iterations...")
                            self.run_benchmark()

                        # Check canvas
                        elif self.canvas_rect.collidepoint(pos):
                            self.canvas.start_draw(pos)
                            self.last_draw_time = time.time()
                            self.benchmark_results = None  # Clear old benchmark when drawing

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.canvas.stop_draw()

                elif event.type == pygame.MOUSEMOTION:
                    if self.canvas.drawing:
                        pos = pygame.mouse.get_pos()
                        if self.canvas_rect.collidepoint(pos):
                            self.canvas.continue_draw(pos)
                            self.last_draw_time = time.time()

            # Auto-inference after drawing stops (only when not paused)
            if self.auto_infer and not self.paused and self.last_draw_time > 0:
                if time.time() - self.last_draw_time > self.infer_delay:
                    self.run_inference()
                    self.last_draw_time = 0

            self.draw()
            self.clock.tick(60)

        pygame.quit()

        # Print final stats
        if self.inference_times:
            avg_time = sum(self.inference_times) / len(self.inference_times)
            print(f"\n[FINAL] Total inferences: {self.total_inferences}")
            print(f"[FINAL] Average inference time: {avg_time:.2f} ms")
            print(f"[FINAL] Throughput: {1000 / avg_time:.0f} inferences/sec")


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "=" * 60)
    print("GENREG MNIST INFERENCE TOOL")
    print("=" * 60)
    print("Draw handwritten digits and see real-time predictions.")

    # Select genome
    genome_info = select_genome()

    # Load genome
    controller, data = load_genome(genome_info)

    print("\n" + "=" * 60)
    print("Starting inference UI...")
    print("=" * 60)

    # Run UI
    ui = MNISTInferenceUI(controller, genome_info)
    ui.run()


if __name__ == "__main__":
    main()
