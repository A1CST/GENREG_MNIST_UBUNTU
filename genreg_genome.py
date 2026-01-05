# ================================================================
# GENREG v2.0 — Genome + Evolution Loop
# ================================================================
# - "Cull & Replace" Evolution
# - Inherited Trust with DECAY (Prevents Inflation)
# - Split Mutation Rates
# - Visual Field: Blank-filling through visual perception
# ================================================================

import random
import copy
import math
from genreg_proteins import run_protein_cascade
from genreg_controller import GENREGController
import config as cfg


# ================================================================
# GENOME CLASS
# ================================================================
class GENREGGenome:
    def __init__(self, proteins, controller):
        self.proteins = proteins            # list[Protein]
        self.controller = controller        # GENREGController
        self.trust = 0.0                    # fitness scalar
        self.trust_history = []             # Last N trust values for variance calculation
        self.stability = 0.0                # Consistency metric (lower variance = higher stability)
        self.food_eaten = 0                 # food eaten in last game (snake)
        self.energy_used = 0.0              # energy consumed in last game (walker)
        self.step_count = 0                 # steps survived in last game
        self.id = random.randint(1000, 9999)

        # ============================================================
        # VISUAL FIELD: State tracking
        # ============================================================
        self.correct_predictions = 0        # Correct blank fills
        self.total_predictions = 0          # Total attempts
        self.consecutive_correct = 0        # Streak tracking
        self.last_word = ""                 # Last word output

        # ============================================================
        # STAGNATION PENALTY: Track repeated wrong answers
        # ============================================================
        self.last_wrong_word = ""           # Last WRONG word output
        self.stagnation_count = 0           # Consecutive same wrong answer

    # ------------------------------------------------------------
    def reset_trust(self):
        self.trust = 0.0

    # ------------------------------------------------------------
    def update_stability(self, window_size=None):
        """Calculate stability as inverse of trust variance over recent episodes."""
        if window_size is None:
            window_size = cfg.STABILITY_WINDOW
        self.trust_history.append(self.trust)

        # Keep only recent history
        if len(self.trust_history) > window_size:
            self.trust_history.pop(0)

        # Need at least 2 episodes to calculate variance
        if len(self.trust_history) < 2:
            self.stability = 0.0
            return

        # Calculate variance
        mean_trust = sum(self.trust_history) / len(self.trust_history)
        variance = sum((t - mean_trust) ** 2 for t in self.trust_history) / len(self.trust_history)

        # Stability = inverse of variance (add 1 to prevent division by zero)
        self.stability = 1.0 / (variance + 1.0)

    # ------------------------------------------------------------
    def reset_visual_state(self):
        """Reset visual-specific state for new episode."""
        self.consecutive_correct = 0
        self.last_word = ""
        # Note: correct_predictions and total_predictions persist for stats
        # Note: stagnation tracking persists across episodes (intentional pressure)

    # ------------------------------------------------------------
    def get_stagnation_multiplier(self, word, is_correct):
        """
        Track repeated wrong answers and return penalty multiplier.

        Thresholds configurable via config.STAGNATION_PENALTIES.
        Default: 3x→1.5x, 5x→2.0x, 10x→3.0x

        Correct answers reset the stagnation counter.
        Different wrong answers reset the counter.

        Returns: multiplier (1.0 = no stagnation, >1.0 = stagnation penalty)
        """
        if is_correct:
            # Correct answer breaks stagnation
            self.last_wrong_word = ""
            self.stagnation_count = 0
            return 1.0

        # Wrong answer - check if same as last wrong
        if word == self.last_wrong_word:
            self.stagnation_count += 1
        else:
            # Different wrong answer - reset counter
            self.last_wrong_word = word
            self.stagnation_count = 1

        # Load thresholds from config
        thresholds = getattr(cfg, 'STAGNATION_PENALTIES', [
            (3, 1.5),
            (5, 2.0),
            (10, 3.0),
        ])

        # Calculate multiplier based on stagnation level
        multiplier = 1.0
        for threshold, mult in thresholds:
            if self.stagnation_count >= threshold:
                multiplier = mult
            else:
                break

        return multiplier

    # ------------------------------------------------------------
    def clone(self):
        """
        Deep copy genome - genetic information only.
        Protein states (perception) reset, but PARAMETERS persist.
        """
        new_genome = GENREGGenome(
            proteins=[copy.deepcopy(p) for p in self.proteins],
            controller=self.controller.clone()
        )

        # NOTE: Trust inheritance is handled in evolve()
        new_genome.trust = 0.0
        new_genome.trust_history = []
        new_genome.stability = 0.0

        # Visual field: reset state
        new_genome.correct_predictions = 0
        new_genome.total_predictions = 0
        new_genome.consecutive_correct = 0
        new_genome.last_word = ""

        # Stagnation tracking: reset
        new_genome.last_wrong_word = ""
        new_genome.stagnation_count = 0

        # Reset internal biological states (perception is individual)
        for p in new_genome.proteins:
            # Reset numeric states to starting values
            for key in p.state:
                if isinstance(p.state[key], (int, float)):
                    if key == "running_mean": p.state[key] = 0.0
                    elif key == "running_max": p.state[key] = 1.0
                    elif key == "count": p.state[key] = 0
                    elif key == "accum": p.state[key] = 0.0
                    elif key == "velocity": p.state[key] = 0.0
                    elif key == "running": p.state[key] = 0.0
                    else: p.state[key] = 0.0
                elif isinstance(p.state[key], bool):
                    p.state[key] = False
                elif p.state[key] is None:
                    p.state[key] = None

        return new_genome

    # ------------------------------------------------------------
    def crossbreed(self, other):
        """
        Create a child genome by mixing two parents.
        Neural network weights: randomly choose from parent1 or parent2
        Protein parameters: average values from both parents
        """
        # Create new controller with mixed weights
        new_controller = self.controller.clone()

        # Check if using PyTorch tensors
        use_torch = hasattr(new_controller, '_use_torch') and new_controller._use_torch

        if use_torch:
            import torch
            # GPU-accelerated crossover using masks
            device = new_controller.device

            # Mix w1: random mask for each weight
            mask_w1 = torch.rand_like(new_controller.w1) < 0.5
            new_controller.w1 = torch.where(mask_w1, other.controller.w1, new_controller.w1)

            # Mix b1
            mask_b1 = torch.rand_like(new_controller.b1) < 0.5
            new_controller.b1 = torch.where(mask_b1, other.controller.b1, new_controller.b1)

            # Mix w2
            mask_w2 = torch.rand_like(new_controller.w2) < 0.5
            new_controller.w2 = torch.where(mask_w2, other.controller.w2, new_controller.w2)

            # Mix b2
            mask_b2 = torch.rand_like(new_controller.b2) < 0.5
            new_controller.b2 = torch.where(mask_b2, other.controller.b2, new_controller.b2)
        else:
            # Pure Python fallback
            # Mix w1: for each weight, randomly pick from parent1 or parent2
            for i in range(len(new_controller.w1)):
                for j in range(len(new_controller.w1[i])):
                    if random.random() < 0.5:
                        new_controller.w1[i][j] = other.controller.w1[i][j]

            # Mix b1
            for i in range(len(new_controller.b1)):
                if random.random() < 0.5:
                    new_controller.b1[i] = other.controller.b1[i]

            # Mix w2
            for i in range(len(new_controller.w2)):
                for j in range(len(new_controller.w2[i])):
                    if random.random() < 0.5:
                        new_controller.w2[i][j] = other.controller.w2[i][j]

            # Mix b2
            for i in range(len(new_controller.b2)):
                if random.random() < 0.5:
                    new_controller.b2[i] = other.controller.b2[i]

        # Create new genome with mixed controller
        child = GENREGGenome(
            proteins=[copy.deepcopy(p) for p in self.proteins],
            controller=new_controller
        )

        # Average protein parameters from both parents
        for i, protein in enumerate(child.proteins):
            for param_name in protein.params:
                parent1_value = self.proteins[i].params[param_name]
                parent2_value = other.proteins[i].params[param_name]
                protein.params[param_name] = (parent1_value + parent2_value) / 2.0

        # Reset biological states (same as clone)
        for p in child.proteins:
            for key in p.state:
                if isinstance(p.state[key], (int, float)):
                    if key == "running_mean": p.state[key] = 0.0
                    elif key == "running_max": p.state[key] = 1.0
                    elif key == "count": p.state[key] = 0
                    elif key == "accum": p.state[key] = 0.0
                    elif key == "velocity": p.state[key] = 0.0
                    elif key == "running": p.state[key] = 0.0
                    else: p.state[key] = 0.0
                elif isinstance(p.state[key], bool):
                    p.state[key] = False
                elif p.state[key] is None:
                    p.state[key] = None

        child.trust = 0.0
        child.trust_history = []
        child.stability = 0.0

        # Visual field: reset state
        child.correct_predictions = 0
        child.total_predictions = 0
        child.consecutive_correct = 0
        child.last_word = ""

        # Stagnation tracking: reset
        child.last_wrong_word = ""
        child.stagnation_count = 0

        return child

    # ------------------------------------------------------------
    def mutate(self, rate=None):
        """Mutate both proteins AND their parameters."""
        if rate is None:
            rate = cfg.MUTATION_RATE
        for p in self.proteins:
            # Mutate hyperparameters with bounds
            for k in p.params:
                if random.random() < rate:
                    p.mutate_param(k, scale=cfg.PROTEIN_MUTATION_SCALE)

                    # Apply parameter bounds from config
                    if k in cfg.PARAM_BOUNDS:
                        min_val, max_val = cfg.PARAM_BOUNDS[k]
                        p.params[k] = max(min(p.params[k], max_val), min_val)

        # Mutate controller weights
        self.controller.mutate(rate)

        return self

    # ------------------------------------------------------------
    def forward(self, signals):
        _, trust_delta = run_protein_cascade(self.proteins, signals)
        self.trust += trust_delta
        # Clamp to prevent overflow
        self.trust = max(min(self.trust, cfg.TRUST_CLAMP_MAX), cfg.TRUST_CLAMP_MIN)
        return trust_delta


# ================================================================
# POPULATION MANAGER (TAX UPDATE)
# ================================================================
class GENREGPopulation:
    def __init__(self, template_proteins, input_size, hidden_size=16, output_size=4, size=20, mutation_rate=0.1):
        self.size = size
        
        # Init population
        self.genomes = []
        for _ in range(size):
            controller = GENREGController(input_size, hidden_size, output_size)
            g = GENREGGenome(
                proteins=[copy.deepcopy(p) for p in template_proteins],
                controller=controller
            )
            self.genomes.append(g)

        self.active = 0

    def get_active(self):
        return self.genomes[self.active]

    def next_genome(self):
        self.active = (self.active + 1) % self.size
        return self.get_active()

    # ------------------------------------------------------------
    # Evolution with Partial Trust Inheritance
    # ------------------------------------------------------------
    def evolve(self):
        """
        Evolution with partial trust inheritance.
        Children inherit baseline competence from proven parents.
        """
        
        # Update stability metrics for all genomes
        for g in self.genomes:
            g.update_stability()

        # Sort by trust (highest first)
        self.genomes.sort(key=lambda g: g.trust, reverse=True)

        # Calculate statistics
        trust_values = [g.trust for g in self.genomes]
        best_trust = trust_values[0]
        median_trust = trust_values[len(trust_values)//2]
        lowest_trust = trust_values[-1]

        stability_values = [g.stability for g in self.genomes]
        best_stability = max(stability_values) if stability_values else 0
        median_stability = sorted(stability_values)[len(stability_values)//2]

        print(f"  > Evolution: Trust[Best={best_trust:.1f} | Med={median_trust:.1f}] Stability[Best={best_stability:.2f}]")

        # Top survivors survive
        survival_cutoff = max(1, int(self.size * cfg.SURVIVAL_CUTOFF))
        survivors = self.genomes[:survival_cutoff]

        # Fitness-proportional weights combining trust and stability
        min_trust = min(g.trust for g in survivors)
        max_stability = max(g.stability for g in survivors) + 1e-6

        fitness_weights = [
            cfg.TRUST_WEIGHT * (g.trust - min_trust + 1.0) + cfg.STABILITY_WEIGHT * (g.stability / max_stability)
            for g in survivors
        ]

        # Replace entire population with offspring
        new_population = []

        for i in range(self.size):
            # Two-parent weighted selection
            parent1, parent2 = random.choices(survivors, weights=fitness_weights, k=2)

            # Crossbreed parents to create child
            child = parent1.crossbreed(parent2)
            child.trust = (parent1.trust + parent2.trust) * cfg.TRUST_INHERITANCE
            child.mutate(rate=cfg.CHILD_MUTATION_RATE)

            new_population.append(child)

        self.genomes = new_population
        self.active = 0
