# ================================================================
# BLANK-FILL VISUAL BOOTSTRAP CONFIGURATION
# ================================================================
# Configuration for fill-in-the-blank visual learning mode.
# The model sees sentences with blanks as pixels and learns to output words.
# ================================================================

# ================================================================
# DEVICE SELECTION
# ================================================================
DEVICE = "auto"  # "auto", "cuda", or "cpu"

# ================================================================
# NEURAL NETWORK
# ================================================================
HIDDEN_SIZE = 16                    # Hidden layer neurons
WEIGHT_INIT_RANGE = (-0.5, 0.5)     # Initial weight range
BIAS_INIT_RANGE = (-0.1, 0.1)       # Initial bias range
MUTATION_RATE = 0.25                # Probability of mutating each weight
MUTATION_SCALE = 0.1                # Gaussian mutation std dev

# ================================================================
# POPULATION & EVOLUTION
# ================================================================
POPULATION_SIZE = 2000                # Number of genomes
CHILD_MUTATION_RATE = 0.08          # Mutation rate for offspring
SURVIVAL_CUTOFF = 0.2               # Top X% survive to reproduce
TRUST_INHERITANCE = 0.03             # Children inherit X% of parent trust

# Selection weights
TRUST_WEIGHT = 0.7                  # Weight for trust in fitness
STABILITY_WEIGHT = 0.3              # Weight for stability in fitness
STABILITY_WINDOW = 5                # Episodes to track for stability

# Trust bounds
TRUST_CLAMP_MIN = -100000.0
TRUST_CLAMP_MAX = 100000.0

# ================================================================
# VISUAL FIELD
# ================================================================
FIELD_WIDTH = 400                   # Pygame window width
FIELD_HEIGHT = 100                  # Pygame window height
FONT_SIZE = 16                      # Text font size in pixels
BLANK_MARKER = "[____]"             # How blanks appear visually

# ================================================================
# CORPUS
# ================================================================
CORPUS_PATH = "corpus/blanks.json"  # Path to fill-in-the-blank sentences

# ================================================================
# NEURAL NETWORK I/O
# ================================================================
INPUT_SIZE = FIELD_WIDTH * FIELD_HEIGHT  # 40000 flattened pixels
# OUTPUT_SIZE is set dynamically based on vocabulary size (loaded from corpus)
# See genreg_visual_env.get_vocabulary()
MAX_WORD_LENGTH = 10                # Maximum word length to generate

# ================================================================
# REWARDS (EXACT MATCH ONLY - matching alphabet bootstrap)
# ================================================================
CORRECT_REWARD = 10.0               # Reward for exact correct answer (same as alphabet)
ALPHABET_WRONG_PENALTY = -1.5       # Penalty for wrong answer (same as alphabet)
STREAK_BONUS = 5.0                  # Bonus per consecutive correct
MAX_STREAK_BONUS = 50.0             # Maximum streak bonus
TRUST_DECAY = 0.02                  # (unused - now using multiplicative decay in script)
EXTRA_LETTER_PENALTY = -5.0         # (unused - now using flat penalty)

# ================================================================
# TRAINING
# ================================================================
CHECKPOINT_INTERVAL = 1000           # Save checkpoint every N generations
CHART_UPDATE_INTERVAL = 10          # Update chart every N generations

# ================================================================
# ALIASES (for compatibility with genreg_visual_env.py)
# ================================================================
VISUAL_FIELD_WIDTH = FIELD_WIDTH
VISUAL_FIELD_HEIGHT = FIELD_HEIGHT
VISUAL_CORRECT_REWARD = CORRECT_REWARD
VISUAL_STREAK_BONUS = STREAK_BONUS
VISUAL_MAX_STREAK_BONUS = MAX_STREAK_BONUS
VISUAL_CLOSENESS_REWARD = 0.0       # No closeness reward - exact match only
VISUAL_FEEDBACK_FRAMES = 10
VISUAL_STEPS_PER_EPISODE = 1        # One attempt per genome per generation

# ================================================================
# ALPHABET RECOGNITION (for AlphabetEnv)
# ================================================================
# Must match config.py values for consistency

# ----- Alphabet-Specific Neural Network -----
ALPHABET_HIDDEN_SIZE = 16           # Hidden layer neurons for alphabet
ALPHABET_POPULATION_SIZE = 200       # Population size for alphabet training
ALPHABET_SEED_COPIES = 1             # Genomes seeded from best genome
ALPHABET_SEED_TRUST = 100.0          # Starting trust for seeded genomes

# ----- Visual Field -----
ALPHABET_FIELD_WIDTH = 100           # Display window width
ALPHABET_FIELD_HEIGHT = 100          # Display window height
ALPHABET_FONT_SIZE = 64              # Font for display window

# ----- VAE Configuration -----
ALPHABET_VAE_PATH = "vae_512_master_best.pt"  # Path to pretrained VAE
ALPHABET_VAE_LATENT_DIM = 128        # VAE latent dimension

# ----- Neural Network I/O -----
ALPHABET_INPUT_SIZE = ALPHABET_FIELD_WIDTH * ALPHABET_FIELD_HEIGHT  # Default: raw pixels
ALPHABET_OUTPUT_SIZE = 26            # 26 letter probabilities (a-z only)

# ----- Reward Structure -----
ALPHABET_CORRECT_REWARD = 10.0       # Reward for correct letter
ALPHABET_WRONG_PENALTY = -1.5        # Penalty for wrong letter

# ----- Training Parameters -----
ALPHABET_LETTERS_PER_EPISODE = 26    # Show all 26 letters per episode
ALPHABET_VARIATIONS_PER_LETTER = 30  # (OLD random mode) Variations per letter
ALPHABET_VIEWS_PER_LETTER = 10       # Times each letter is shown
ALPHABET_RANDOMIZE_ORDER = True      # Shuffle letter order each episode
ALPHABET_CHECKPOINT_INTERVAL = 500   # Save checkpoint every N generations

# ----- Variation Ranges -----
ALPHABET_SIZE_RANGE = (0.5, 1.5)     # Font size multiplier range
ALPHABET_ROTATION_RANGE = (-25, 25)  # Rotation in degrees
ALPHABET_POSITION_RANGE = (-15, 15)  # Position offset in pixels
ALPHABET_MIN_CONTRAST = 100          # Minimum brightness difference
