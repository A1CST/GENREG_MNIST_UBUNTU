# ================================================================
# GENREG Configuration
# ================================================================
# All tunable parameters in one place.
# Supports Visual Field, Walker2D, and Language Bootstrap modes.
# ================================================================


# ================================================================
# MODE SELECTION
# ================================================================
MODE = "alphabet"  # "mnist", "alphabet", "visual", "walker", or "language"


# ================================================================
# DEVICE SELECTION (GPU/CPU)
# ================================================================
# Options: "auto", "cuda", "cpu"
# "auto" will use CUDA if available, otherwise CPU
DEVICE = "auto"


# ================================================================
# SHARED NEURAL NETWORK DEFAULTS
# ================================================================
# These are defaults - each mode can override with its own settings
HIDDEN_SIZE = 64                 # Default hidden layer neurons
WEIGHT_INIT_RANGE = (-0.5, 0.5)     # Initial weight range
BIAS_INIT_RANGE = (-0.1, 0.1)       # Initial bias range
MUTATION_SCALE = 0.1                # Gaussian mutation std dev


# ================================================================
# SHARED POPULATION & EVOLUTION DEFAULTS
# ================================================================
# These are defaults - each mode can override with its own settings
POPULATION_SIZE = 2000               # Default population size
MUTATION_RATE = 0.12                # Probability of mutating each weight
CHILD_MUTATION_RATE = 0.15          # Mutation rate for offspring
SURVIVAL_CUTOFF = 0.2               # Top X% survive to reproduce
TRUST_INHERITANCE = 0.5            # Children inherit X% of parent trust (0.5 = average of parents)

# Selection weights
TRUST_WEIGHT = 0.7                  # Weight for trust in fitness
STABILITY_WEIGHT = 0.3              # Weight for stability in fitness
STABILITY_WINDOW = 5                # Episodes to track for stability calc


# ================================================================
# TRUST
# ================================================================
TRUST_CLAMP_MIN = -100000.0         # Minimum trust value
TRUST_CLAMP_MAX = 100000.0          # Maximum trust value


# ================================================================
# PROTEIN MUTATION BOUNDS
# ================================================================
PROTEIN_MUTATION_SCALE = 0.2        # Scale for protein param mutation

# Parameter-specific bounds
PARAM_BOUNDS = {
    "scale": (-5.0, 5.0),
    "gain": (0.1, 10.0),
    "decay": (0.0, 0.999),
    "threshold": (-10.0, 10.0),
    "momentum": (0.0, 0.99),
    "countdown_max": (1, 20),        # Language: max countdown steps
}


# ================================================================
# ENVIRONMENT (Walker2D)
# ================================================================
MAX_EPISODE_STEPS = 100000            # Steps before episode ends
FRAME_SKIP = 4                      # Physics steps per action
HEALTHY_Z_RANGE = (0.8, 2.0)        # Valid torso height range
HEALTHY_ANGLE_RANGE = (-1.0, 1.0)   # Valid torso angle range
RESET_NOISE_SCALE = 5e-3            # Noise added on reset


# ================================================================
# VISUAL FIELD CONFIGURATION
# ================================================================
# Pygame window dimensions (small window, no downsampling)
VISUAL_FIELD_WIDTH = 400            # Pygame window width
VISUAL_FIELD_HEIGHT = 100           # Pygame window height
FONT_SIZE = 16                      # Text font size in pixels

# Visual markers
BLANK_MARKER = "[____]"             # How blanks appear in the visual field

# Corpus
CORPUS_PATH = "corpus/blanks.json"  # Path to fill-in-the-blank sentences

# Neural network I/O for visual mode
# Input: flattened grayscale pixels (400 * 100 = 40000)
VISUAL_INPUT_SIZE = VISUAL_FIELD_WIDTH * VISUAL_FIELD_HEIGHT
VISUAL_OUTPUT_SIZE = 27             # 27 character probabilities (a-z + space)
MAX_WORD_LENGTH = 10                # Maximum word length model can output

# Reward structure (POSITIVE ONLY - no penalties)
VISUAL_CORRECT_REWARD = 50.0        # Big reward for exact correct answer
VISUAL_CLOSENESS_REWARD = 10.0      # Max reward for closeness (scaled by match %)
VISUAL_STREAK_BONUS = 5.0           # Bonus per consecutive correct
VISUAL_MAX_STREAK_BONUS = 50.0      # Maximum streak bonus
VISUAL_TRUST_DECAY = 0.1           # Trust decays by this amount per step

# Visual effects
VISUAL_FEEDBACK_FRAMES = 10         # Frames to show visual feedback

# Episode settings
VISUAL_STEPS_PER_EPISODE = 50       # Steps per genome per generation


# ================================================================
# ALPHABET RECOGNITION CONFIGURATION
# ================================================================
# Simplest possible task: see a letter, output that letter
# Tests: Can model distinguish visual patterns? Can it learn exact mappings?
# STATUS: COMPLETED - 100% SUCCESS at generation 1862

# ----- Alphabet-Specific Neural Network -----
ALPHABET_HIDDEN_SIZE = 128           # Hidden layer neurons for alphabet
ALPHABET_POPULATION_SIZE = 200       # Population size for alphabet training
ALPHABET_SEED_COPIES = 1            # Genomes seeded from best genome (when using [S] option)
ALPHABET_SEED_TRUST = 100.0          # Starting trust for seeded genomes

# ----- Visual Field -----
ALPHABET_FIELD_WIDTH = 100           # Display window width
ALPHABET_FIELD_HEIGHT = 100          # Display window height
ALPHABET_FONT_SIZE = 64              # Font for display window

# ----- VAE Configuration (use --vae flag to enable) -----
ALPHABET_VAE_PATH = "vae_512_master_best.pt"  # Path to pretrained VAE
ALPHABET_VAE_LATENT_DIM = 128        # VAE latent dimension

# ----- Neural Network I/O -----
# Input size is set dynamically in bootstrap script based on --vae flag
# With --vae: 128-dim latent | Without: 10000 pixels (100x100)
ALPHABET_INPUT_SIZE = ALPHABET_FIELD_WIDTH * ALPHABET_FIELD_HEIGHT  # Default: raw pixels
ALPHABET_OUTPUT_SIZE = 26            # 26 letter probabilities (a-z only, no space)

# ----- Reward Structure (BINARY, no partial credit) -----
ALPHABET_CORRECT_REWARD = 10.0       # Reward for correct letter
ALPHABET_WRONG_PENALTY = -1.5        # Penalty for wrong letter

# ----- Training Parameters -----
ALPHABET_LETTERS_PER_EPISODE = 26    # Show all 26 letters per episode
ALPHABET_VARIATIONS_PER_LETTER = 30  # (OLD random mode) Variations per letter
ALPHABET_VIEWS_PER_LETTER = 10       # Times each letter is shown (cycles through 4 fixed variations)
ALPHABET_RANDOMIZE_ORDER = True      # Shuffle letter order each episode
ALPHABET_CHECKPOINT_INTERVAL = 500   # Save checkpoint every N generations

# ----- Variation Ranges -----
ALPHABET_SIZE_RANGE = (0.5, 1.5)     # Font size multiplier range
ALPHABET_ROTATION_RANGE = (-25, 25)  # Rotation in degrees
ALPHABET_POSITION_RANGE = (-15, 15)  # Position offset in pixels
ALPHABET_MIN_CONTRAST = 100          # Minimum brightness difference between letter and background


# ================================================================
# MNIST DIGIT RECOGNITION CONFIGURATION
# ================================================================
# Handwritten digit recognition using the official MNIST benchmark
# Tests: Can model learn real-world visual patterns (not pygame-rendered)?
# STATUS: In progress

# ----- MNIST-Specific Neural Network -----
MNIST_HIDDEN_SIZE = 16               # Hidden layer neurons for MNIST
MNIST_POPULATION_SIZE = 500          # Population size for MNIST training

# ----- Visual Field -----
MNIST_FIELD_WIDTH = 28               # MNIST native width
MNIST_FIELD_HEIGHT = 28              # MNIST native height

# ----- Neural Network I/O -----
# Input: 28x28 = 784 pixels (native MNIST resolution)
MNIST_INPUT_SIZE = MNIST_FIELD_WIDTH * MNIST_FIELD_HEIGHT  # 784
MNIST_OUTPUT_SIZE = 10               # 10 digit probabilities (0-9)

# ----- Reward Structure (BINARY, no partial credit) -----
MNIST_CORRECT_REWARD = 10.0          # Reward for correct digit
MNIST_WRONG_PENALTY = -2.5           # Penalty for wrong digit

# ----- Training Parameters -----
MNIST_DIGITS_PER_EPISODE = 10        # Show all 10 digits per episode
MNIST_IMAGES_PER_DIGIT = 20          # Images shown per digit per generation (more = stable signal)
MNIST_RANDOMIZE_ORDER = True         # Shuffle digit order each episode
MNIST_CHECKPOINT_INTERVAL = 500     # Save checkpoint every N generations


# ================================================================
# LANGUAGE BOOTSTRAP CONFIGURATION (ARCHIVED)
# ================================================================

# Countdown mechanism
INITIAL_COUNTDOWN_MAX = 5           # Starting max countdown (evolves) - start small
COUNTDOWN_MIN = 1                   # Minimum countdown steps
COUNTDOWN_MAX = 20                  # Maximum countdown steps

# Text generation
VOCAB_SIZE = 27                     # a-z + space
MAX_OUTPUT_LENGTH = 50              # Max characters per output
TEXT_EMBEDDING_SIZE = 32            # Embedding dimension for text

# Fitness evaluation
WORD_DETECTION_MIN_LENGTH = 2       # Minimum word length to count
WORD_DETECTION_MAX_LENGTH = 10      # Maximum word length to count

# Novelty system
NOVELTY_DECAY = 0.95                # How fast novelty diminishes for repeated words
NOVELTY_BASE_REWARD =50.0          # Base reward for valid word
NOVELTY_BONUS_NEW_WORD = 5.0        # Bonus for first occurrence of a word

# Random encouragement (prevents silence)
RANDOM_ENCOURAGEMENT_PROB = 0.07    # 7% chance of random positive feedback
RANDOM_ENCOURAGEMENT_REWARD = 3.0   # Reward amount for random encouragement

# Linguistic environment
SENTENCE_NOISE_PROB = 0.1           # Probability of adding noise to input
INPUT_CONTEXT_LENGTH = 200          # Max characters of linguistic context

# Phase 1: Multi-word output rewards
# Goal: Reward outputs with MORE real words, penalize gibberish
MULTI_WORD_BONUS = 5.0              # Bonus per additional valid word (increased)
MULTI_WORD_SCALING = 1.3            # Exponential scaling: bonus * (scaling ^ word_count)
WORD_RATIO_BONUS = 30.0             # Max bonus for high valid/total word ratio
MIN_WORD_RATIO_FOR_BONUS = 0.5      # Minimum ratio to get any ratio bonus
GIBBERISH_PENALTY = -5            # Penalty per gibberish token
PERFECT_RATIO_THRESHOLD = 0.8       # Ratio considered "perfect" (all/mostly real words)

# Phase 2+: Semantic bonuses
BIGRAM_BONUS = 1.5                  # Bonus for valid word pairs


# ================================================================
# PHASE 2: MASKED SEMANTIC PREDICTION
# ================================================================

# Phase transition triggers
PHASE2_TRIGGER_SUCCESS_RATE = 0.80  # Trigger Phase 2 after 80%+ Phase 1 success
PHASE2_TRIGGER_MIN_VOCAB = 50       # Minimum vocabulary size required
PHASE2_TRIGGER_MIN_GENERATIONS = 50 # Minimum generations in Phase 1

# Fitness rewards
PHASE2_CATEGORY_MATCH_REWARD = 50.0 # Exact category match (primary goal)
PHASE2_VALID_WORD_REWARD = 10.0     # Valid word, wrong category
PHASE2_ANY_CATEGORY_BONUS = 5.0     # Word is in some semantic category
PHASE2_LENGTH_CREDIT = 2.0          # Correct length for category
PHASE2_PREFIX_CREDIT = 1.0          # Shares prefix with valid words
PHASE2_NOVELTY_BONUS = 3.0          # First-time prediction
PHASE2_EMPTY_PENALTY = -5.0         # Empty/too-short output

# Difficulty progression
PHASE2_INITIAL_DIFFICULTY = 1       # Start with difficulty 1 templates
PHASE2_DIFFICULTY_INCREASE_RATE = 50  # Increase difficulty every N generations
PHASE2_MAX_DIFFICULTY = 3           # Maximum difficulty level

# Training parameters
PHASE2_TEMPLATES_PER_EPISODE = 5    # Templates to evaluate per genome per generation
PHASE2_SUCCESS_THRESHOLD = 0.70     # Target success rate to complete Phase 2


# ================================================================
# TRAINING LOOP
# ================================================================
MAX_WORKERS = 16                    # Parallel environments
CHECKPOINT_INTERVAL = 1000             # Save every N generations (more frequent for language)
STATS_INTERVAL = 50                 # Detailed stats every N generations
CHART_UPDATE_INTERVAL = 5           # Update charts every N generations


# ================================================================
# OUTPUT ACTIONS
# ================================================================
OUTPUT_SIZE = 6                     # Walker2D: 6 joint torques
LANGUAGE_OUTPUT_SIZE = 29           # Language: 27 chars + countdown + string_length
