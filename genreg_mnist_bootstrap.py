# ================================================================
# GENREG MNIST Bootstrap - Handwritten Digit Recognition
# ================================================================
# Benchmark task: Recognize handwritten digits from the MNIST dataset
#
# Input: 28x28 grayscale image (784 pixels) of handwritten digit
# Output: That digit (0-9)
# Reward: +10 correct, -1.5 wrong (no partial credit)
#
# Usage:
#   python genreg_mnist_bootstrap.py
#
# This tests:
# - Can the model learn real-world visual patterns (not pygame-rendered)?
# - Can evolutionary learning match gradient-based methods on MNIST?
# - Is 784-dimensional input tractable for the architecture?
# ================================================================

import time
import os
import sys
import random
from datetime import datetime
from collections import deque

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import config as cfg
from genreg_visual_env import MNISTEnv, PYGAME_AVAILABLE
from genreg_controller import GENREGController, BatchedController, TORCH_AVAILABLE, DEVICE
from genreg_genome import GENREGPopulation, GENREGGenome
from genreg_checkpoint import save_checkpoint, get_latest_checkpoint, load_checkpoint

if PYGAME_AVAILABLE:
    import pygame

# Multiprocessing for charts (avoid pygame/matplotlib GIL conflict)
import multiprocessing as mp

# Fix Windows multiprocessing spawn issues
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)


# ================================================================
# MNIST-SPECIFIC TRAINING PARAMETERS
# ================================================================
# Trust decay per step - prevents runaway accumulation, keeps fitness meaningful
# With 10 digits × 20 images = 200 steps/gen, 0.001 gives ~18% total decay/gen
MNIST_TRUST_DECAY = 0.001  # 0.1% decay each image evaluation


# ================================================================
# CHART PROCESS (runs in separate process)
# ================================================================
def _chart_process(data_queue, stop_event):
    """Run matplotlib charts in a separate process."""
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[CHARTS] Failed to initialize matplotlib: {e}")
        return

    # Enable interactive mode
    plt.ion()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GENREG MNIST Recognition - Training Progress', fontsize=14)

    ax_success = axes[0, 0]
    ax_fitness = axes[0, 1]
    ax_mastered = axes[1, 0]
    ax_time = axes[1, 1]

    # Data storage
    generations = []
    success_rates = []
    rolling_success = []
    fitness_best = []
    fitness_median = []
    mastered_count = []
    gen_times = []

    # Initialize plots
    ax_success.set_title('Success Rate')
    ax_success.set_xlabel('Generation')
    ax_success.set_ylabel('Accuracy (%)')
    ax_success.set_ylim(0, 100)
    ax_success.grid(True, alpha=0.3)
    line_success, = ax_success.plot([], [], 'b-', label='Per-Gen', alpha=0.5)
    line_rolling, = ax_success.plot([], [], 'b-', linewidth=2, label='Rolling Avg')
    ax_success.axhline(y=10, color='r', linestyle='--', alpha=0.3, label='Random (10%)')
    ax_success.legend(loc='upper left', fontsize=8)

    ax_fitness.set_title('Fitness (Trust)')
    ax_fitness.set_xlabel('Generation')
    ax_fitness.set_ylabel('Trust')
    ax_fitness.grid(True, alpha=0.3)
    line_fit_best, = ax_fitness.plot([], [], 'g-', linewidth=2, label='Best')
    line_fit_med, = ax_fitness.plot([], [], 'orange', linewidth=1, label='Median')
    ax_fitness.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax_fitness.legend(loc='upper left', fontsize=8)

    ax_mastered.set_title('Digits Mastered (>90% accuracy)')
    ax_mastered.set_xlabel('Generation')
    ax_mastered.set_ylabel('Count')
    ax_mastered.set_ylim(0, 10)
    ax_mastered.grid(True, alpha=0.3)
    line_mastered, = ax_mastered.plot([], [], 'purple', linewidth=2)
    ax_mastered.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Goal (10)')
    ax_mastered.legend(loc='upper left', fontsize=8)

    ax_time.set_title('Generation Time')
    ax_time.set_xlabel('Generation')
    ax_time.set_ylabel('Seconds')
    ax_time.grid(True, alpha=0.3)
    line_time, = ax_time.plot([], [], 'red', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.show(block=False)

    # Main loop
    while not stop_event.is_set():
        # Check for new data
        try:
            while not data_queue.empty():
                data = data_queue.get_nowait()
                generations.append(data['gen'])
                success_rates.append(data['success_rate'] * 100)
                rolling_success.append(data['rolling_avg'] * 100)
                fitness_best.append(data['best_fitness'])
                fitness_median.append(data['median_fitness'])
                mastered_count.append(data['mastered'])
                gen_times.append(data['gen_time'])

            # Update plots if we have data
            if generations:
                line_success.set_data(generations, success_rates)
                line_rolling.set_data(generations, rolling_success)
                line_fit_best.set_data(generations, fitness_best)
                line_fit_med.set_data(generations, fitness_median)
                line_mastered.set_data(generations, mastered_count)
                line_time.set_data(generations, gen_times)

                xmax = max(10, max(generations) + 1)
                for ax in axes.flat:
                    ax.set_xlim(0, xmax)

                if fitness_best:
                    ymin = min(min(fitness_median), 0) * 1.1 if fitness_median else -10
                    ymax = max(fitness_best) * 1.1 if max(fitness_best) > 0 else 10
                    ax_fitness.set_ylim(ymin, ymax)

                if gen_times:
                    ax_time.set_ylim(0, max(gen_times) * 1.2)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.1)

        except Exception as e:
            pass

    plt.close(fig)


# ================================================================
# REAL-TIME CHART CONTROLLER
# ================================================================
class RealtimeCharts:
    """Controller for real-time charts running in separate process."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.data_queue = None
        self.stop_event = None
        self.process = None

        if not enabled:
            print("[CHARTS] Charts disabled")
            return

        try:
            self.data_queue = mp.Queue()
            self.stop_event = mp.Event()
            self.process = mp.Process(target=_chart_process, args=(self.data_queue, self.stop_event))
            self.process.daemon = True  # Don't block exit
            self.process.start()

            # Give it a moment to start
            time.sleep(0.5)

            if self.process.is_alive():
                print("[CHARTS] Started real-time chart window (separate process)")
            else:
                print("[CHARTS] Chart process failed to start, continuing without charts")
                self.enabled = False
        except Exception as e:
            print(f"[CHARTS] Failed to start chart process: {e}")
            self.enabled = False

    def update(self, gen, success_rate, rolling_avg, best_fitness, median_fitness,
               mastered, gen_time):
        """Send data to chart process."""
        if not self.enabled or self.data_queue is None:
            return

        try:
            self.data_queue.put_nowait({
                'gen': gen,
                'success_rate': success_rate,
                'rolling_avg': rolling_avg,
                'best_fitness': best_fitness,
                'median_fitness': median_fitness,
                'mastered': mastered,
                'gen_time': gen_time
            })
        except:
            pass  # Queue full or process dead, ignore

    def close(self):
        """Stop the chart process."""
        if not self.enabled or self.process is None:
            return

        try:
            self.stop_event.set()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
            print("[CHARTS] Closed chart window")
        except:
            pass


# ================================================================
# MNIST STATS TRACKER
# ================================================================
class MNISTStatsTracker:
    """Track MNIST digit recognition statistics."""

    def __init__(self, window_size=20):
        self.window_size = window_size

        # Rolling windows
        self.success_rates = deque(maxlen=window_size)
        self.avg_fitness = deque(maxlen=window_size)
        self.best_fitness = deque(maxlen=window_size)
        self.gen_times = deque(maxlen=window_size)

        # Per-digit accuracy tracking
        self.per_digit_correct = {str(i): 0 for i in range(10)}
        self.per_digit_attempts = {str(i): 0 for i in range(10)}

        # Cumulative stats
        self.total_correct = 0
        self.total_attempts = 0
        self.total_generations = 0
        self.start_time = time.time()

        # Peak values
        self.peak_success_rate = 0.0
        self.peak_fitness = float('-inf')

        # Digits mastered (>90% accuracy)
        self.mastered_digits = set()

    def record_generation(self, gen_num, correct, attempts, fitness_list, gen_time,
                          per_digit_results):
        """Record stats for one generation."""
        self.total_generations = gen_num
        self.total_correct += correct
        self.total_attempts += attempts

        # Calculate metrics
        success_rate = correct / max(1, attempts)
        avg_fit = sum(fitness_list) / max(1, len(fitness_list))
        best_fit = max(fitness_list) if fitness_list else 0

        # Add to rolling windows
        self.success_rates.append(success_rate)
        self.avg_fitness.append(avg_fit)
        self.best_fitness.append(best_fit)
        self.gen_times.append(gen_time)

        # Update peaks
        self.peak_success_rate = max(self.peak_success_rate, success_rate)
        self.peak_fitness = max(self.peak_fitness, best_fit)

        # Update per-digit stats
        for digit, (c, a) in per_digit_results.items():
            self.per_digit_correct[digit] += c
            self.per_digit_attempts[digit] += a

        # Check for mastered digits
        for digit in self.per_digit_correct:
            attempts = self.per_digit_attempts[digit]
            if attempts >= 10:  # Need at least 10 attempts
                accuracy = self.per_digit_correct[digit] / attempts
                if accuracy >= 0.9:
                    self.mastered_digits.add(digit)

    def get_rolling_avg(self, window):
        """Get average of a rolling window."""
        if not window:
            return 0.0
        return sum(window) / len(window)

    def format_generation_log(self, gen_num, correct, attempts, best_genome, gen_time):
        """Format generation log line."""
        success_rate = correct / max(1, attempts)
        roll_success = self.get_rolling_avg(self.success_rates)
        roll_fitness = self.get_rolling_avg(self.avg_fitness)

        lines = []

        # Main status line
        status = (f"[GEN {gen_num:>5}] "
                  f"Correct: {correct:>2}/{attempts:<2} ({success_rate:>5.1%}) | "
                  f"Fitness: {best_genome.trust:>7.1f} | "
                  f"Time: {gen_time:>4.1f}s")
        lines.append(status)

        # Rolling averages
        roll_line = (f"           "
                     f"Rolling({self.window_size}): "
                     f"Success={roll_success:>5.1%} | "
                     f"Fitness={roll_fitness:>7.1f} | "
                     f"Mastered: {len(self.mastered_digits)}/10")
        lines.append(roll_line)

        return "\n".join(lines)

    def format_digit_breakdown(self):
        """Format per-digit accuracy breakdown."""
        lines = ["", "  Per-Digit Accuracy:"]

        row = "    "
        for digit in "0123456789":
            attempts = self.per_digit_attempts[digit]
            if attempts > 0:
                acc = self.per_digit_correct[digit] / attempts
                if acc >= 0.9:
                    row += f"{digit}:OK  "
                elif acc >= 0.5:
                    row += f"{digit}:{acc:.0%} "
                else:
                    row += f"{digit}:-- "
            else:
                row += f"{digit}:?  "
        lines.append(row)

        return "\n".join(lines)

    def format_summary(self, gen_num):
        """Format detailed summary."""
        elapsed = time.time() - self.start_time
        gens_per_min = gen_num / max(1, elapsed / 60)
        overall_success = self.total_correct / max(1, self.total_attempts)

        lines = [
            "",
            "=" * 70,
            f"SUMMARY @ Generation {gen_num}",
            "=" * 70,
            f"  Runtime: {elapsed/60:.1f} min ({gens_per_min:.1f} gen/min)",
            f"  Overall: {self.total_correct}/{self.total_attempts} correct ({overall_success:.1%})",
            "",
            f"  Peak Success Rate: {self.peak_success_rate:.1%}",
            f"  Peak Fitness:      {self.peak_fitness:.1f}",
            "",
            f"  Digits Mastered (>90%): {len(self.mastered_digits)}/10",
        ]

        if self.mastered_digits:
            lines.append(f"    {' '.join(sorted(self.mastered_digits))}")

        # Show struggling digits
        struggling = []
        for digit in "0123456789":
            attempts = self.per_digit_attempts[digit]
            if attempts >= 10:
                acc = self.per_digit_correct[digit] / attempts
                if acc < 0.5:
                    struggling.append(f"{digit}({acc:.0%})")

        if struggling:
            lines.append("")
            lines.append(f"  Struggling Digits (<50%): {', '.join(struggling)}")

        lines.append("")
        lines.append(self.format_digit_breakdown())
        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)


# ================================================================
# SESSION MANAGEMENT
# ================================================================
def prepare_session_config():
    """Prepare session configuration and checkpoint directory."""
    base_dir = "checkpoints_mnist"
    os.makedirs(base_dir, exist_ok=True)

    # List existing sessions
    sessions = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            sessions.append(name)

    sessions.sort(reverse=True)

    print("\n" + "=" * 60)
    print("SESSION MANAGEMENT")
    print("=" * 60)

    if sessions:
        print("\nExisting sessions:")
        for i, session in enumerate(sessions[:5]):
            checkpoint = get_latest_checkpoint(os.path.join(base_dir, session))
            if checkpoint:
                print(f"  [{i+1}] {session} (has checkpoints)")
            else:
                print(f"  [{i+1}] {session} (no checkpoints)")

        print("\nOptions:")
        print("  [N] Start NEW session")
        print("  [1-5] Resume existing session")
        print("  [Enter] Start new session (default)")

        choice = input("\nYour choice: ").strip().upper()

        if choice.isdigit() and 1 <= int(choice) <= len(sessions):
            session_name = sessions[int(choice) - 1]
            checkpoint_dir = os.path.join(base_dir, session_name)
            checkpoint_path = get_latest_checkpoint(checkpoint_dir)
            print(f"\nResuming session: {session_name}")
            return {
                "checkpoint_path": checkpoint_path,
                "checkpoint_dir": checkpoint_dir,
                "session_name": session_name,
            }

    # Create new session
    timestamp = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nStarting NEW session: {timestamp}")
    return {
        "checkpoint_path": None,
        "checkpoint_dir": checkpoint_dir,
        "session_name": timestamp,
    }


# ================================================================
# MAIN TRAINING LOOP
# ================================================================
def mnist_bootstrap_loop(session_config):
    """Main training loop for MNIST digit recognition."""
    checkpoint_path = session_config.get("checkpoint_path")
    checkpoint_dir = session_config.get("checkpoint_dir")

    # Network dimensions for MNIST mode
    input_size = cfg.MNIST_INPUT_SIZE  # 784 (28x28)
    hidden_size = cfg.MNIST_HIDDEN_SIZE  # MNIST-specific hidden size
    output_size = cfg.MNIST_OUTPUT_SIZE  # 10 digits

    print(f"\n[MNIST] Using {input_size} pixel input (28x28)")

    # Initialize or load population
    generation = 0
    if checkpoint_path:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        try:
            population, generation, _, _ = load_checkpoint(checkpoint_path)
            print(f"Resumed from generation {generation}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh...")
            population = None
    else:
        population = None

    if population is None:
        population = GENREGPopulation(
            template_proteins=[],  # No proteins needed for simple digit task
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            size=cfg.MNIST_POPULATION_SIZE,
        )

    pop_size = population.size

    # Initialize real-time charts BEFORE pygame (avoids TkAgg conflict)
    charts = RealtimeCharts()

    # Create MNIST environment (initializes pygame)
    env = MNISTEnv(render_mode="human", device=str(DEVICE) if DEVICE else "cpu")

    # Initialize stats tracker
    stats = MNISTStatsTracker(window_size=20)

    print(f"\n{'=' * 70}")
    print("GENREG MNIST DIGIT RECOGNITION")
    print("=" * 70)
    print(f"  Population:   {pop_size} genomes")
    print(f"  Input size:   {input_size} pixels (28x28 MNIST)")
    print(f"  Output size:  {output_size} (digit probs 0-9)")
    print(f"  Hidden size:  {hidden_size}")
    print(f"  GPU:          {'Yes (' + str(DEVICE) + ')' if TORCH_AVAILABLE and DEVICE else 'No (CPU)'}")
    print("=" * 70)
    print("\nTask: See a handwritten digit, output that digit.")
    print("Reward: +10 correct, -1.5 wrong (no partial credit)")
    print("=" * 70)
    print("\nControls:")
    print("  ESC or close window: Stop training")
    print("  Space: Pause/Resume")
    print("=" * 70)
    print("\nStarting training...", flush=True)
    evals_per_gen = pop_size * 10 * cfg.MNIST_IMAGES_PER_DIGIT
    print(f"Each generation: {pop_size} genomes x 10 digits x {cfg.MNIST_IMAGES_PER_DIGIT} images = {evals_per_gen:,} evaluations\n", flush=True)

    # Training state
    running = True
    paused = False
    generation_start_time = time.time()
    summary_interval = 10

    # Create batched controller for GPU-accelerated evaluation
    if TORCH_AVAILABLE and DEVICE:
        batched = BatchedController([g.controller for g in population.genomes], device=DEVICE)
        print(f"[BATCH] Using BatchedController for {pop_size} genomes (10 GPU calls/gen)")
    else:
        batched = None
        print("[BATCH] BatchedController not available, using sequential evaluation")

    try:
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"\n{'=' * 30} {'PAUSED' if paused else 'RESUMED'} {'=' * 30}\n")

            if paused:
                env.clock.tick(10)
                continue

            # ========================================
            # EVALUATE ALL GENOMES (BATCHED)
            # ========================================
            gen_correct = 0
            gen_attempts = 0
            per_digit_results = {str(i): [0, 0] for i in range(10)}  # [correct, attempts]

            # Reset visual state (but NOT trust - trust persists across generations)
            for genome in population.genomes:
                genome.reset_visual_state()

            # Shuffle digit order for this generation
            digit_order = list(range(10))
            if cfg.MNIST_RANDOMIZE_ORDER:
                random.shuffle(digit_order)

            # BATCHED: For each digit, show multiple images and evaluate ALL genomes
            for digit_pos, digit_idx in enumerate(digit_order):
                expected = str(digit_idx)

                # Show multiple images of this digit for more stable signal
                for image_num in range(cfg.MNIST_IMAGES_PER_DIGIT):
                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            running = False

                    if not running:
                        break

                    # Select a NEW random image of this digit
                    env._select_digit(digit_idx)
                    obs = env.get_observation()

                    # Render the digit
                    env.render()

                    # Progress indicator
                    print(f"\r  Digit {digit_pos+1}/10 ({expected}), Image {image_num+1}/{cfg.MNIST_IMAGES_PER_DIGIT}...", end="", flush=True)

                    if batched:
                        # ONE GPU call: Get predictions from ALL genomes for this image
                        digits = batched.generate_digit_all(obs)
                    else:
                        # Fallback: Sequential evaluation
                        digits = [g.controller.generate_digit(obs) for g in population.genomes]

                    # Evaluate all predictions
                    for genome_idx, (genome, digit) in enumerate(zip(population.genomes, digits)):
                        is_correct = (digit == expected)

                        gen_attempts += 1
                        per_digit_results[expected][1] += 1

                        # Apply trust decay at each step (prevents runaway accumulation)
                        genome.trust *= (1.0 - MNIST_TRUST_DECAY)

                        if is_correct:
                            gen_correct += 1
                            per_digit_results[expected][0] += 1
                            genome.trust += cfg.MNIST_CORRECT_REWARD
                        else:
                            genome.trust += cfg.MNIST_WRONG_PENALTY

                if not running:
                    break

            if not running:
                break

            # Collect fitness values
            genome_fitness = [g.trust for g in population.genomes]

            # ========================================
            # RECORD STATISTICS
            # ========================================
            gen_time = time.time() - generation_start_time

            stats.record_generation(
                gen_num=generation,
                correct=gen_correct,
                attempts=gen_attempts,
                fitness_list=genome_fitness,
                gen_time=gen_time,
                per_digit_results=per_digit_results
            )

            # ========================================
            # LOGGING
            # ========================================
            print()  # Clear progress line
            best_genome = max(population.genomes, key=lambda g: g.trust)

            log = stats.format_generation_log(
                gen_num=generation,
                correct=gen_correct,
                attempts=gen_attempts,
                best_genome=best_genome,
                gen_time=gen_time
            )
            print(log, flush=True)

            # Update real-time charts
            success_rate = gen_correct / max(1, gen_attempts)
            rolling_avg = stats.get_rolling_avg(stats.success_rates)
            sorted_fitness = sorted(genome_fitness)
            median_fitness = sorted_fitness[len(sorted_fitness) // 2] if sorted_fitness else 0
            charts.update(
                gen=generation,
                success_rate=success_rate,
                rolling_avg=rolling_avg,
                best_fitness=best_genome.trust,
                median_fitness=median_fitness,
                mastered=len(stats.mastered_digits),
                gen_time=gen_time
            )

            # Periodic summary
            if generation > 0 and generation % summary_interval == 0:
                summary = stats.format_summary(generation)
                print(summary)

            # Check for completion (all 10 digits mastered)
            if len(stats.mastered_digits) == 10:
                print("\n" + "=" * 70)
                print("MNIST COMPLETE! All 10 digits mastered!")
                print("=" * 70)
                break

            # ========================================
            # CHECKPOINT (BEFORE evolution to preserve trust)
            # ========================================
            if generation > 0 and generation % cfg.MNIST_CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    population, generation, [],
                    checkpoint_dir=checkpoint_dir
                )
                print(f"  >>> CHECKPOINT saved (gen {generation}) [trust preserved]\n")

            # ========================================
            # EVOLUTION
            # ========================================
            population.evolve()
            generation += 1
            generation_start_time = time.time()

            # Sync batched controller with new weights after evolution
            if batched:
                batched.sync_from_controllers([g.controller for g in population.genomes])

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("INTERRUPTED - Saving checkpoint...")
        print("=" * 70)

    finally:
        # Final summary
        print(stats.format_summary(generation))

        # Close charts
        charts.close()

        # Save final checkpoint
        save_checkpoint(
            population, generation, [],
            checkpoint_dir=checkpoint_dir
        )
        print(f"[FINAL] Saved checkpoint at generation {generation}")

        # Cleanup
        env.close()

    return population, generation


# ================================================================
# ENTRY POINT
# ================================================================
def main():
    """Main entry point."""
    if not PYGAME_AVAILABLE:
        print("ERROR: Pygame is required for MNIST mode")
        print("Install with: pip install pygame")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GENREG MNIST DIGIT RECOGNITION")
    print("=" * 70)
    print("Handwritten digit recognition using the official MNIST benchmark.")
    print()
    print("Input:  28x28 grayscale image (784 pixels)")
    print("Output: Digit (0-9)")
    print("Reward: +10 correct, -1.5 wrong")
    print()
    print("This tests:")
    print("  - Can the model learn real-world visual patterns?")
    print("  - Can evolutionary learning work on MNIST?")
    print("  - Is 784-dimensional input tractable?")
    print("=" * 70)

    session_config = prepare_session_config()
    population, final_gen = mnist_bootstrap_loop(session_config)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Final generation: {final_gen}")
    print(f"  Session: {session_config['session_name']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
