# ================================================================
# GENREG Visual Field Environment
# ================================================================
# Evolutionary Learning Through Visual Perception
#
# Core concept: Agent perceives a 2D visual field where text is
# rendered as pixels. Blanks appear as gaps [____] that the agent
# must fill. Correct predictions cause world evolution; incorrect
# predictions cause degradation or stasis.
#
# Gradient-based reward: Wrong answers get scaled penalties based
# on how close they are to any valid answer. This creates a smooth
# fitness landscape that evolution can navigate.
# ================================================================

import json
import random
import os

import blank_config as cfg

# Try to import pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[VISUAL ENV] Pygame not available - visual mode disabled")

# Try to import numpy for array operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[VISUAL ENV] NumPy not available - using pure Python")


# ================================================================
# CORPUS LOADING
# ================================================================
def load_corpus(path=None):
    """Load fill-in-the-blank corpus from JSON file."""
    if path is None:
        path = cfg.CORPUS_PATH

    # Handle relative paths
    if not os.path.isabs(path):
        # Try relative to current directory
        if not os.path.exists(path):
            # Try relative to this file's directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, path)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("sentences", [])
    except FileNotFoundError:
        print(f"[VISUAL ENV] Corpus not found at {path}, using fallback")
        return _fallback_corpus()
    except json.JSONDecodeError:
        print(f"[VISUAL ENV] Invalid JSON in {path}, using fallback")
        return _fallback_corpus()


def _fallback_corpus():
    """Fallback corpus if file not found."""
    return [
        {"text": "the cat sat on the ____", "answers": ["mat", "bed", "rug"]},
        {"text": "the ____ is blue", "answers": ["sky", "sea", "car"]},
        {"text": "a dog can ____", "answers": ["run", "sit", "dig"]},
    ]


def extract_vocabulary(corpus=None):
    """
    Extract all unique words from corpus answers.
    Returns sorted list of vocabulary words.
    """
    if corpus is None:
        corpus = load_corpus()

    vocab = set()
    for sentence in corpus:
        for answer in sentence.get("answers", []):
            vocab.add(answer.lower().strip())

    return sorted(vocab)


# Global vocabulary (loaded once)
VOCABULARY = None

def get_vocabulary():
    """Get or load the vocabulary list."""
    global VOCABULARY
    if VOCABULARY is None:
        VOCABULARY = extract_vocabulary()
        print(f"[VOCAB] Loaded {len(VOCABULARY)} words: {VOCABULARY[:10]}...")
    return VOCABULARY


# ================================================================
# GRADIENT-BASED REWARD CALCULATION
# ================================================================
def longest_common_subsequence(s1, s2):
    """
    Calculate longest common subsequence length between two strings.
    This measures characters that appear in the same ORDER (not contiguous).

    Example:
        "ski" vs "sky" -> LCS = 2 ("sk")
        "sik" vs "sky" -> LCS = 1 ("s") - 'i' breaks the order
        "xyz" vs "sky" -> LCS = 0
    """
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0

    # Dynamic programming table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def calculate_closeness(output_word, valid_answers):
    """
    Calculate how close an output word is to the nearest valid answer.

    Uses longest common subsequence with TWO penalties:
    1. Base penalty: LCS / max(len(output), len(answer))
    2. Extra letter penalty: -5% per incorrect/extra character (configurable)

    This means:
        - "bed" vs "bed" = 100% (perfect match)
        - "be" vs "bed" = 67% - 5% = 62% (missing one, one wrong)
        - "bedd" vs "bed" = 75% - 5% = 70% (one extra char)
        - "hatter" vs "hat" = 50% - 15% = 35% (3 extra chars)
        - "xgpcclbeed" vs "bed" = 30% - 35% = 0% (7 extra chars, clamped)

    Returns:
        tuple: (best_score, best_match, normalized_score)
        - best_score: raw LCS length
        - best_match: which answer was closest
        - normalized_score: LCS / max_length with extra letter penalty (0.0 to 1.0)
    """
    if not output_word or not valid_answers:
        return 0, None, 0.0

    # Get extra letter penalty from config
    extra_penalty = getattr(cfg, 'EXTRA_LETTER_PENALTY', 0.05)

    best_normalized = 0.0
    best_score = 0
    best_match = valid_answers[0]

    for answer in valid_answers:
        lcs = longest_common_subsequence(output_word, answer)
        # Penalize for length mismatch - divide by the LONGER string
        max_len = max(len(output_word), len(answer))
        normalized = lcs / max_len if max_len > 0 else 0.0

        # Additional penalty for extra incorrect letters
        # Extra letters = characters in output that aren't part of the LCS
        extra_letters = len(output_word) - lcs
        extra_letter_penalty = extra_letters * extra_penalty
        normalized = max(0.0, normalized - extra_letter_penalty)

        if normalized > best_normalized:
            best_normalized = normalized
            best_score = lcs
            best_match = answer

    return best_score, best_match, best_normalized


# ================================================================
# VISUAL FIELD ENVIRONMENT
# ================================================================
class VisualFieldEnv:
    """
    Pygame-based visual environment for GENREG.

    Provides:
    - Visual field rendering (text as pixels)
    - Blank detection and filling
    - Visual feedback for correct/incorrect actions
    - World evolution based on performance
    """

    def __init__(self, render_mode="human"):
        """
        Initialize the visual field.

        Args:
            render_mode: "human" (display window) or "headless" (no display)
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is required for VisualFieldEnv")

        self.render_mode = render_mode
        self.width = cfg.VISUAL_FIELD_WIDTH
        self.height = cfg.VISUAL_FIELD_HEIGHT

        # Initialize pygame
        pygame.init()

        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("GENREG Visual Field")
        else:
            # Headless mode - use a surface
            self.screen = pygame.Surface((self.width, self.height))

        self.font = pygame.font.Font(None, cfg.FONT_SIZE)
        self.clock = pygame.time.Clock()

        # Load corpus
        self.corpus = load_corpus()
        if not self.corpus:
            raise RuntimeError("No corpus loaded - cannot initialize environment")

        # World state
        self.current_sentence = None
        self.current_answers = []
        self.world_health = 1.0
        self.step_count = 0
        self.consecutive_correct = 0

        # Visual effects
        self.effect_frames_remaining = 0
        self.current_effect = None

        # Cached observation (reused until sentence changes)
        self.cached_observation = None
        self.observation_dirty = True

        # Statistics
        self.total_correct = 0
        self.total_attempts = 0

    def reset(self):
        """Reset environment for new episode."""
        self._select_new_sentence()
        self.world_health = 1.0
        self.step_count = 0
        self.consecutive_correct = 0
        self.effect_frames_remaining = 0
        self.current_effect = None
        return self.get_observation()

    def _select_new_sentence(self):
        """Select a new sentence with blank from corpus."""
        entry = random.choice(self.corpus)
        self.current_sentence = entry["text"]
        self.current_answers = [a.lower() for a in entry["answers"]]
        self.observation_dirty = True  # Need to re-render

    def _render_field(self):
        """Render the current visual field."""
        # Background color based on world health (clamp to valid range)
        bg_intensity = max(0, min(255, int(self.world_health * 40)))
        bg_blue = max(0, min(255, bg_intensity + 10))
        self.screen.fill((bg_intensity, bg_intensity, bg_blue))

        # Prepare display text with blank marker
        display_text = self.current_sentence.replace("____", cfg.BLANK_MARKER)

        # Determine text color based on effects
        if self.effect_frames_remaining > 0:
            if self.current_effect == "correct":
                text_color = (100, 255, 100)  # Green flash
            elif self.current_effect == "wrong":
                text_color = (255, 100, 100)  # Red tint
            else:
                text_color = (255, 255, 255)
            self.effect_frames_remaining -= 1
        else:
            text_color = (255, 255, 255)

        # Render text centered
        text_surface = self.font.render(display_text, True, text_color)
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text_surface, text_rect)

        # Update display if in human mode
        if self.render_mode == "human":
            pygame.display.flip()

    def get_observation(self):
        """
        Get current visual observation for the agent.

        Returns:
            list of floats: flattened grayscale pixel values normalized to [0, 1]
        """
        # Render if observation is dirty
        if self.observation_dirty:
            self._render_field()
            self._cache_observation()
            self.observation_dirty = False

        return self.cached_observation

    def _cache_observation(self):
        """Cache the current screen as a normalized pixel array."""
        if NUMPY_AVAILABLE:
            # Fast numpy-based capture
            pixels = pygame.surfarray.array3d(self.screen)
            # Convert to grayscale: average of RGB channels
            grayscale = np.mean(pixels, axis=2)
            # Normalize to [0, 1]
            normalized = grayscale / 255.0
            # Flatten
            self.cached_observation = normalized.flatten().tolist()
        else:
            # Pure Python fallback (slower)
            self.cached_observation = []
            for y in range(self.height):
                for x in range(self.width):
                    pixel = self.screen.get_at((x, y))
                    # Grayscale = average of RGB
                    gray = (pixel.r + pixel.g + pixel.b) / 3.0 / 255.0
                    self.cached_observation.append(gray)

    def step(self, word):
        """
        Execute action (word) and return observation, reward, done, info.

        Args:
            word: string - the word to fill the blank with

        Returns:
            observation: visual field pixels (list of floats)
            reward: float fitness score
            done: bool episode ended
            info: dict with details
        """
        self.step_count += 1
        self.total_attempts += 1

        # Normalize the word
        word = word.lower().strip()

        # Check if word is correct
        is_correct = word in self.current_answers

        if is_correct:
            reward = self._calculate_correct_reward()
            self.total_correct += 1
            self.consecutive_correct += 1
            self.current_effect = "correct"
            self.effect_frames_remaining = cfg.VISUAL_FEEDBACK_FRAMES

            # World evolves: new sentence appears, health restored
            self._select_new_sentence()
            self.world_health = 1.0  # Fresh start for new sentence

        else:
            # Calculate closeness (rewards handled by bootstrap script)
            reward, closeness_info = self._calculate_closeness(word)
            self.consecutive_correct = 0
            self.current_effect = "wrong"
            self.effect_frames_remaining = cfg.VISUAL_FEEDBACK_FRAMES
            # Sentence stays the same until correct answer

        # Get new observation
        observation = self.get_observation()

        # Episode ends if world health depleted or max steps reached
        done = self.world_health <= 0 or self.step_count >= cfg.VISUAL_STEPS_PER_EPISODE

        info = {
            "is_correct": is_correct,
            "word": word,
            "expected": self.current_answers,
            "world_health": self.world_health,
            "consecutive_correct": self.consecutive_correct,
            "step": self.step_count,
            "sentence": self.current_sentence,
        }

        # Add closeness info for wrong answers
        if not is_correct:
            info["closeness"] = closeness_info

        return observation, reward, done, info

    def _calculate_correct_reward(self):
        """Calculate reward for correct prediction."""
        base_reward = cfg.VISUAL_CORRECT_REWARD
        streak_bonus = min(
            self.consecutive_correct * cfg.VISUAL_STREAK_BONUS,
            cfg.VISUAL_MAX_STREAK_BONUS
        )
        return base_reward + streak_bonus

    def _calculate_closeness(self, word):
        """
        Calculate closeness to valid answers (no penalty, positive rewards only).

        Returns closeness info that the bootstrap script uses to calculate rewards.
        Closer matches = higher rewards (handled by bootstrap script).

        Returns:
            tuple: (reward, closeness_info)
        """
        # Calculate closeness to nearest valid answer
        lcs_score, closest_match, normalized_score = calculate_closeness(
            word, self.current_answers
        )

        # Closeness info for reward calculation in bootstrap script
        closeness_info = {
            "lcs_score": lcs_score,
            "closest_match": closest_match,
            "normalized_score": normalized_score,
        }

        # Return 0 reward - bootstrap script calculates positive rewards based on closeness
        return 0.0, closeness_info

    def get_signals(self):
        """
        Get environment signals for protein cascade.

        Returns dict compatible with run_protein_cascade.
        """
        return {
            "world_health": self.world_health,
            "consecutive_correct": float(self.consecutive_correct),
            "step_count": float(self.step_count) / cfg.VISUAL_STEPS_PER_EPISODE,
            "accuracy": self.total_correct / max(1, self.total_attempts),
        }

    def render(self):
        """Force render the current state (for visualization)."""
        self._render_field()
        if self.render_mode == "human":
            self.clock.tick(30)  # Cap at 30 FPS

    def close(self):
        """Cleanup pygame resources."""
        pygame.quit()


# ================================================================
# ALPHABET RECOGNITION ENVIRONMENT (STAGE 1) - VAE VERSION
# ================================================================
class AlphabetEnv:
    """
    Simplest possible visual learning task with VAE encoding.

    Shows a single letter (A-Z), model must output that letter.
    Binary reward: correct = +10, wrong = -1.5
    No partial credit, no context, pure pattern → symbol mapping.

    Uses pretrained VAE encoder to compress 512x512 image to 128-dim latent.
    """

    def __init__(self, render_mode="human", vae_encoder=None, device="cuda"):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is required for AlphabetEnv")

        self.render_mode = render_mode
        self.vae_encoder = vae_encoder
        self.device = device
        self.use_vae = vae_encoder is not None

        # Display window (small for viewing)
        self.display_width = cfg.ALPHABET_FIELD_WIDTH
        self.display_height = cfg.ALPHABET_FIELD_HEIGHT

        # VAE input size (512x512 for the pretrained model)
        self.vae_width = 512
        self.vae_height = 512

        # Initialize pygame
        pygame.init()

        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            pygame.display.set_caption("GENREG Alphabet Recognition (VAE)")
        else:
            self.screen = pygame.Surface((self.display_width, self.display_height))

        # High-res surface for VAE input
        self.vae_surface = pygame.Surface((self.vae_width, self.vae_height))

        # Use a larger font for the VAE surface
        self.display_font = pygame.font.Font(None, cfg.ALPHABET_FONT_SIZE)
        self.vae_font = pygame.font.Font(None, 300)  # Large font for 512x512

        self.clock = pygame.time.Clock()

        # All 26 letters
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # Current state
        self.current_letter = None
        self.current_letter_idx = 0
        self.letter_order = list(range(26))
        self.variation = None  # Variation parameters (size, rotation, position, color)

        # Statistics
        self.total_correct = 0
        self.total_attempts = 0
        self.per_letter_correct = {letter: 0 for letter in self.letters}
        self.per_letter_attempts = {letter: 0 for letter in self.letters}

        # Cached observation
        self.cached_observation = None
        self.observation_dirty = True

        # Import torch if using VAE
        if self.use_vae:
            import torch
            self.torch = torch

    def reset(self):
        """Reset for new episode - shuffle letter order."""
        if cfg.ALPHABET_RANDOMIZE_ORDER:
            random.shuffle(self.letter_order)
        self.current_letter_idx = 0
        self._select_letter(self.letter_order[0])
        return self.get_observation()

    def _select_letter(self, letter_idx):
        """Select a letter to display (no variation, centered)."""
        self.current_letter = self.letters[letter_idx]
        self.variation = None  # No variation
        self.observation_dirty = True

    def _select_letter_with_variation(self, letter_idx):
        """Select a letter with random augmentation (size, rotation, position, color, background)."""
        self.current_letter = self.letters[letter_idx]

        # Letter color: alternate between white and black
        is_white_letter = random.choice([True, False])
        if is_white_letter:
            letter_color = (255, 255, 255)
        else:
            letter_color = (0, 0, 0)

        # Generate random background color (RGB)
        # Ensure sufficient contrast with letter
        min_contrast = cfg.ALPHABET_MIN_CONTRAST
        while True:
            bg_r = random.randint(0, 255)
            bg_g = random.randint(0, 255)
            bg_b = random.randint(0, 255)

            # Calculate brightness (perceived luminance)
            bg_brightness = 0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b
            letter_brightness = 255 if is_white_letter else 0

            # Check contrast
            if abs(bg_brightness - letter_brightness) >= min_contrast:
                break

        bg_color = (bg_r, bg_g, bg_b)

        # Generate random variation parameters
        self.variation = {
            'size_mult': random.uniform(cfg.ALPHABET_SIZE_RANGE[0], cfg.ALPHABET_SIZE_RANGE[1]),
            'rotation': random.uniform(cfg.ALPHABET_ROTATION_RANGE[0], cfg.ALPHABET_ROTATION_RANGE[1]),
            'offset_x': random.randint(cfg.ALPHABET_POSITION_RANGE[0], cfg.ALPHABET_POSITION_RANGE[1]),
            'offset_y': random.randint(cfg.ALPHABET_POSITION_RANGE[0], cfg.ALPHABET_POSITION_RANGE[1]),
            'letter_color': letter_color,
            'bg_color': bg_color
        }
        self.observation_dirty = True

    def _select_letter_with_fixed_variation(self, letter_idx, variation_idx):
        """
        Select a letter with one of 4 fixed variations (no randomness).

        Variations:
            0: 12pt font, white text on black background
            1: 12pt font, black text on white background
            2: 64pt font (normal), white text on black background
            3: 64pt font (normal), black text on white background
        """
        self.current_letter = self.letters[letter_idx]

        # Define the 4 fixed variations
        # (size_mult, letter_color, bg_color)
        # size_mult of 0.1875 gives 12pt from 64pt base (12/64 = 0.1875)
        SMALL_SIZE = 12 / cfg.ALPHABET_FONT_SIZE  # 12pt
        NORMAL_SIZE = 1.0  # 64pt (normal)

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        fixed_variations = [
            (SMALL_SIZE, WHITE, BLACK),   # 0: 12pt, white on black
            (SMALL_SIZE, BLACK, WHITE),   # 1: 12pt, black on white
            (NORMAL_SIZE, WHITE, BLACK),  # 2: 64pt, white on black
            (NORMAL_SIZE, BLACK, WHITE),  # 3: 64pt, black on white
        ]

        size_mult, letter_color, bg_color = fixed_variations[variation_idx % 4]

        self.variation = {
            'size_mult': size_mult,
            'rotation': 0,      # No rotation
            'offset_x': 0,      # Centered
            'offset_y': 0,      # Centered
            'letter_color': letter_color,
            'bg_color': bg_color
        }
        self.observation_dirty = True

    def _render_field(self):
        """Render the current letter on both display and VAE surfaces."""
        # Get variation parameters (or defaults)
        if self.variation:
            size_mult = self.variation['size_mult']
            rotation = self.variation['rotation']
            offset_x = self.variation['offset_x']
            offset_y = self.variation['offset_y']
            letter_color = self.variation['letter_color']
            bg_color = self.variation['bg_color']
        else:
            size_mult = 1.0
            rotation = 0
            offset_x = 0
            offset_y = 0
            letter_color = (255, 255, 255)
            bg_color = (0, 0, 0)

        # Create font with varied size for display
        varied_font_size = max(16, int(cfg.ALPHABET_FONT_SIZE * size_mult))
        varied_display_font = pygame.font.Font(None, varied_font_size)

        # Render display surface (small, for human viewing)
        self.screen.fill(bg_color)
        text_surface = varied_display_font.render(self.current_letter, True, letter_color)

        # Apply rotation if needed
        if rotation != 0:
            text_surface = pygame.transform.rotate(text_surface, rotation)

        # Position with offset
        center_x = self.display_width // 2 + offset_x
        center_y = self.display_height // 2 + offset_y
        text_rect = text_surface.get_rect(center=(center_x, center_y))
        self.screen.blit(text_surface, text_rect)

        # Render VAE surface (512x512, for encoding) with same variations
        if self.use_vae:
            varied_vae_size = max(50, int(300 * size_mult))
            varied_vae_font = pygame.font.Font(None, varied_vae_size)

            self.vae_surface.fill(bg_color)
            vae_text = varied_vae_font.render(self.current_letter, True, letter_color)

            if rotation != 0:
                vae_text = pygame.transform.rotate(vae_text, rotation)

            # Scale offsets for VAE surface (5.12x larger)
            vae_offset_x = int(offset_x * 5.12)
            vae_offset_y = int(offset_y * 5.12)
            vae_center_x = self.vae_width // 2 + vae_offset_x
            vae_center_y = self.vae_height // 2 + vae_offset_y
            vae_rect = vae_text.get_rect(center=(vae_center_x, vae_center_y))
            self.vae_surface.blit(vae_text, vae_rect)

        if self.render_mode == "human":
            pygame.display.flip()

    def get_observation(self):
        """Get current observation (VAE latent or raw pixels)."""
        if self.observation_dirty:
            self._render_field()
            self._cache_observation()
            self.observation_dirty = False
        return self.cached_observation

    def _cache_observation(self):
        """Cache observation - either VAE latent or raw pixels."""
        if self.use_vae:
            self._cache_vae_latent()
        else:
            self._cache_raw_pixels()

    def _cache_vae_latent(self):
        """Encode the VAE surface to latent vector."""
        # Get RGB pixels from VAE surface
        pixels = pygame.surfarray.array3d(self.vae_surface)  # (W, H, 3)
        pixels = np.transpose(pixels, (2, 0, 1))  # (3, W, H) - note: pygame is W,H not H,W
        pixels = np.transpose(pixels, (0, 2, 1))  # (3, H, W) - fix orientation
        pixels = pixels.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert to tensor
        tensor = self.torch.from_numpy(pixels).unsqueeze(0).to(self.device)  # (1, 3, 512, 512)

        # Encode
        with self.torch.no_grad():
            latent = self.vae_encoder(tensor)  # (1, 128)

        # Convert to list
        self.cached_observation = latent.squeeze(0).cpu().tolist()

    def _cache_raw_pixels(self):
        """Cache raw pixels (fallback if no VAE)."""
        if NUMPY_AVAILABLE:
            pixels = pygame.surfarray.array3d(self.screen)
            grayscale = np.mean(pixels, axis=2)
            normalized = grayscale / 255.0
            self.cached_observation = normalized.flatten().tolist()
        else:
            self.cached_observation = []
            for y in range(self.display_height):
                for x in range(self.display_width):
                    pixel = self.screen.get_at((x, y))
                    gray = (pixel.r + pixel.g + pixel.b) / 3.0 / 255.0
                    self.cached_observation.append(gray)

    def step(self, char):
        """
        Evaluate single character output.

        Args:
            char: single character (a-z or A-Z)

        Returns:
            observation, reward, done, info
        """
        self.total_attempts += 1
        self.per_letter_attempts[self.current_letter] += 1

        # Normalize to uppercase for comparison
        char = char.upper().strip()
        if len(char) > 0:
            char = char[0]
        else:
            char = ""

        # Binary evaluation
        is_correct = (char == self.current_letter)

        if is_correct:
            reward = cfg.ALPHABET_CORRECT_REWARD
            self.total_correct += 1
            self.per_letter_correct[self.current_letter] += 1
        else:
            reward = cfg.ALPHABET_WRONG_PENALTY

        # Move to next letter
        self.current_letter_idx += 1
        done = self.current_letter_idx >= cfg.ALPHABET_LETTERS_PER_EPISODE

        if not done:
            next_letter_idx = self.letter_order[self.current_letter_idx % 26]
            self._select_letter(next_letter_idx)

        info = {
            "is_correct": is_correct,
            "expected": self.current_letter if not done else self.letters[self.letter_order[(self.current_letter_idx - 1) % 26]],
            "output": char,
            "letter_idx": self.current_letter_idx - 1,
        }

        # Get new observation (next letter)
        observation = self.get_observation() if not done else self.cached_observation

        return observation, reward, done, info

    def get_accuracy_by_letter(self):
        """Get accuracy breakdown by letter."""
        result = {}
        for letter in self.letters:
            attempts = self.per_letter_attempts[letter]
            correct = self.per_letter_correct[letter]
            result[letter] = correct / max(1, attempts)
        return result

    def get_signals(self):
        """Get environment signals for protein cascade."""
        return {
            "accuracy": self.total_correct / max(1, self.total_attempts),
            "letter_progress": self.current_letter_idx / 26.0,
        }

    def render(self):
        """Force render current state."""
        self._render_field()
        if self.render_mode == "human":
            self.clock.tick(30)

    def close(self):
        """Cleanup pygame."""
        pygame.quit()


# ================================================================
# MNIST DIGIT RECOGNITION ENVIRONMENT
# ================================================================
class MNISTEnv:
    """
    MNIST handwritten digit recognition environment.

    Uses the official MNIST benchmark dataset (60,000 training images).
    Shows a single handwritten digit (0-9), model must output that digit.
    Binary reward: correct = +10, wrong = -1.5
    No partial credit, pure pattern → digit mapping.

    Downloads MNIST automatically via torchvision if not present.

    GPU OPTIMIZATION: Pre-loads all images to GPU for fast batched access.
    """

    def __init__(self, render_mode="human", device="cuda"):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is required for MNISTEnv")

        self.render_mode = render_mode
        self.device = device

        # Display window (100x100 for viewing, MNIST is 28x28)
        self.display_width = 100
        self.display_height = 100

        # MNIST native size
        self.mnist_width = 28
        self.mnist_height = 28

        # Initialize pygame
        pygame.init()

        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            pygame.display.set_caption("GENREG MNIST Recognition")
        else:
            self.screen = pygame.Surface((self.display_width, self.display_height))

        self.clock = pygame.time.Clock()

        # Load MNIST dataset
        print("[MNIST] Loading MNIST dataset...")
        self._load_mnist()
        print(f"[MNIST] Loaded {len(self.mnist_data)} training images")

        # All 10 digits
        self.digits = list("0123456789")

        # Organize images by digit for balanced sampling
        self._organize_by_digit()

        # GPU OPTIMIZATION: Pre-load all images to GPU
        self._preload_to_gpu()

        # Current state
        self.current_digit = None
        self.current_digit_idx = 0
        self.current_image = None
        self.current_image_idx = None  # Track which image index for GPU lookup
        self.digit_order = list(range(10))

        # Statistics
        self.total_correct = 0
        self.total_attempts = 0
        self.per_digit_correct = {digit: 0 for digit in self.digits}
        self.per_digit_attempts = {digit: 0 for digit in self.digits}

        # Cached observation
        self.cached_observation = None
        self.observation_dirty = True

    def _load_mnist(self):
        """Download and load MNIST dataset."""
        try:
            from torchvision import datasets, transforms
            import torch
            self.torch = torch

            # Download MNIST to ./data directory
            self.mnist_data = datasets.MNIST(
                './data',
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
        except ImportError as e:
            raise RuntimeError(
                "torchvision is required for MNIST. Install with: pip install torchvision"
            ) from e

    def _organize_by_digit(self):
        """Organize MNIST images by digit label for balanced sampling."""
        self.images_by_digit = {i: [] for i in range(10)}

        for idx in range(len(self.mnist_data)):
            image, label = self.mnist_data[idx]
            self.images_by_digit[label].append(idx)

        # Print distribution
        for digit in range(10):
            count = len(self.images_by_digit[digit])
            print(f"  Digit {digit}: {count} images")

    def _preload_to_gpu(self):
        """Pre-load ALL MNIST images to GPU as a single tensor for fast access."""
        print(f"[MNIST] Pre-loading all images to {self.device}...")

        n_images = len(self.mnist_data)

        # Stack all images into a single tensor (N, 784) on GPU
        all_images = []
        for idx in range(n_images):
            image, _ = self.mnist_data[idx]
            all_images.append(image.view(-1))  # Flatten to 784

        # Stack and move to GPU
        self.gpu_images = self.torch.stack(all_images).to(self.device)

        # Also create per-digit GPU tensors for even faster sampling
        self.gpu_images_by_digit = {}
        for digit in range(10):
            indices = self.images_by_digit[digit]
            self.gpu_images_by_digit[digit] = self.gpu_images[indices]

        print(f"[MNIST] GPU tensor shape: {self.gpu_images.shape} ({self.gpu_images.element_size() * self.gpu_images.nelement() / 1024 / 1024:.1f} MB)")

    def reset(self):
        """Reset for new episode - shuffle digit order."""
        random.shuffle(self.digit_order)
        self.current_digit_idx = 0
        self._select_digit(self.digit_order[0])
        return self.get_observation()

    def _select_digit(self, digit_idx):
        """Select a random MNIST image of the given digit."""
        self.current_digit = str(digit_idx)

        # Pick a random image index within this digit's images
        n_images = len(self.images_by_digit[digit_idx])
        local_idx = random.randint(0, n_images - 1)

        # Store for GPU lookup
        self.current_digit_int = digit_idx
        self.current_local_idx = local_idx

        # Get from GPU tensor for observation (stays on GPU until needed)
        self.current_image = self.gpu_images_by_digit[digit_idx][local_idx].cpu().numpy().reshape(28, 28)

        self.observation_dirty = True

    def get_observation_gpu(self):
        """Get current observation as GPU tensor (no CPU conversion)."""
        return self.gpu_images_by_digit[self.current_digit_int][self.current_local_idx]

    def get_batch_observations_gpu(self, digit_idx, n_images):
        """
        Get multiple random images of a digit directly as GPU tensor.

        Args:
            digit_idx: which digit (0-9)
            n_images: how many images to sample

        Returns:
            torch.Tensor of shape (n_images, 784) on GPU
        """
        digit_images = self.gpu_images_by_digit[digit_idx]
        n_available = digit_images.shape[0]

        # Random sample indices
        indices = self.torch.randint(0, n_available, (n_images,), device=self.device)

        return digit_images[indices]

    # ================================================================
    # AUGMENTATION METHODS
    # ================================================================

    def _aug_random_resized_crop(self, image):
        """
        Random Resized Crop - crop 50-80% of image and resize back to 28x28.
        Forces model to recognize digits from partial views.

        Args:
            image: tensor of shape (784,) - flattened 28x28 image

        Returns:
            Augmented image tensor of shape (784,)
        """
        img_2d = image.view(28, 28)

        # Random crop size (50-80% of original)
        crop_ratio = random.uniform(0.5, 0.8)
        crop_size = int(28 * crop_ratio)

        # Random crop position
        max_offset = 28 - crop_size
        crop_x = random.randint(0, max_offset)
        crop_y = random.randint(0, max_offset)

        # Extract crop
        crop = img_2d[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

        # Resize back to 28x28 using nearest neighbor (simple but fast)
        # Use repeat to upscale
        scale = 28 / crop_size
        if scale > 1:
            # Upscale using interpolation
            crop_np = crop.cpu().numpy()
            # Simple bilinear-ish resize using numpy
            y_indices = np.clip((np.arange(28) / scale).astype(int), 0, crop_size - 1)
            x_indices = np.clip((np.arange(28) / scale).astype(int), 0, crop_size - 1)
            resized = crop_np[np.ix_(y_indices, x_indices)]
            result = self.torch.from_numpy(resized).float().to(self.device)
        else:
            result = crop

        return result.view(784)

    def _aug_random_affine(self, image):
        """
        Random Affine - rotation (±15-30°) and translation (10-20%).
        Creates spatial invariance.

        Args:
            image: tensor of shape (784,) - flattened 28x28 image

        Returns:
            Augmented image tensor of shape (784,)
        """
        img_2d = image.view(28, 28).cpu().numpy()

        # Random rotation (±15 to ±30 degrees, but not too extreme for digits)
        angle = random.choice([-1, 1]) * random.uniform(10, 25)

        # Random translation (10-20% of image size = 2-5 pixels)
        tx = random.randint(-4, 4)
        ty = random.randint(-4, 4)

        # Create rotation matrix
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Center of image
        cx, cy = 14, 14

        # Apply affine transform
        result = np.zeros((28, 28), dtype=np.float32)

        for y in range(28):
            for x in range(28):
                # Translate to center, rotate, translate back, then apply shift
                src_x = cos_a * (x - cx) + sin_a * (y - cy) + cx - tx
                src_y = -sin_a * (x - cx) + cos_a * (y - cy) + cy - ty

                # Bilinear interpolation
                x0, y0 = int(np.floor(src_x)), int(np.floor(src_y))
                x1, y1 = x0 + 1, y0 + 1

                if 0 <= x0 < 27 and 0 <= y0 < 27:
                    fx, fy = src_x - x0, src_y - y0
                    result[y, x] = (
                        img_2d[y0, x0] * (1 - fx) * (1 - fy) +
                        img_2d[y0, x1] * fx * (1 - fy) +
                        img_2d[y1, x0] * (1 - fx) * fy +
                        img_2d[y1, x1] * fx * fy
                    )
                elif 0 <= x0 < 28 and 0 <= y0 < 28:
                    result[y, x] = img_2d[y0, x0]

        return self.torch.from_numpy(result).float().to(self.device).view(784)

    def _aug_gaussian_blur_noise(self, image):
        """
        Gaussian Blur or Noise - removes high-frequency details.
        Forces model to learn shape, not texture artifacts.

        Args:
            image: tensor of shape (784,) - flattened 28x28 image

        Returns:
            Augmented image tensor of shape (784,)
        """
        img_2d = image.view(28, 28)

        # Randomly choose blur or noise
        if random.random() < 0.5:
            # Gaussian blur using simple 3x3 kernel
            img_np = img_2d.cpu().numpy()
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

            # Pad and convolve
            padded = np.pad(img_np, 1, mode='constant', constant_values=0)
            result = np.zeros_like(img_np)

            for y in range(28):
                for x in range(28):
                    result[y, x] = np.sum(padded[y:y+3, x:x+3] * kernel)

            return self.torch.from_numpy(result).float().to(self.device).view(784)
        else:
            # Add Gaussian noise
            noise_std = random.uniform(0.05, 0.15)
            noise = self.torch.randn_like(image) * noise_std
            noisy = (image + noise).clamp(0, 1)
            return noisy

    def _apply_random_augmentation(self, image):
        """
        Apply a randomly selected augmentation to a single image.

        Args:
            image: tensor of shape (784,)

        Returns:
            Augmented image tensor of shape (784,)
        """
        aug_type = random.randint(0, 2)

        if aug_type == 0:
            return self._aug_random_resized_crop(image)
        elif aug_type == 1:
            return self._aug_random_affine(image)
        else:
            return self._aug_gaussian_blur_noise(image)

    def prepare_generation_batch(self, images_per_digit, shift_ratio=0.5, max_shift=3):
        """
        Pre-select all images for an entire generation with augmentation.

        Augmentation strategy:
        - 10 images (half): shifted by a few pixels (simple augmentation)
        - 5 images (quarter): random heavy augmentation (crop/affine/blur)
        - 5 images (quarter): unmodified originals

        Args:
            images_per_digit: number of images per digit
            shift_ratio: fraction of images to shift (default 0.5 = half)
            max_shift: maximum pixels to shift in any direction (default 3)

        Returns:
            dict mapping digit -> tensor of shape (images_per_digit, 784) on GPU
        """
        batch = {}
        n_heavy_aug = 5  # 5 images get heavy augmentation (crop/affine/blur)
        n_shift = int(images_per_digit * shift_ratio) - n_heavy_aug  # Remaining shifted

        for digit in range(10):
            images = self.get_batch_observations_gpu(digit, images_per_digit)

            # Shuffle indices to randomly select which images get which augmentation
            all_indices = list(range(images_per_digit))
            random.shuffle(all_indices)

            # First n_heavy_aug get heavy augmentation
            heavy_indices = all_indices[:n_heavy_aug]
            # Next n_shift get simple shift
            shift_indices = all_indices[n_heavy_aug:n_heavy_aug + n_shift]
            # Rest stay unmodified

            # Apply heavy augmentations (crop/affine/blur)
            for idx in heavy_indices:
                images[idx] = self._apply_random_augmentation(images[idx])

            # Apply simple shifts
            for idx in shift_indices:
                shift_x = random.randint(-max_shift, max_shift)
                shift_y = random.randint(-max_shift, max_shift)
                if shift_x != 0 or shift_y != 0:
                    img_2d = images[idx].view(28, 28)
                    shifted = self.torch.zeros_like(img_2d)

                    src_x_start = max(0, -shift_x)
                    src_x_end = min(28, 28 - shift_x)
                    dst_x_start = max(0, shift_x)
                    dst_x_end = min(28, 28 + shift_x)
                    src_y_start = max(0, -shift_y)
                    src_y_end = min(28, 28 - shift_y)
                    dst_y_start = max(0, shift_y)
                    dst_y_end = min(28, 28 + shift_y)

                    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        img_2d[src_y_start:src_y_end, src_x_start:src_x_end]

                    images[idx] = shifted.view(784)

            batch[digit] = images
        return batch

    def _render_field(self):
        """Render the current MNIST digit on the display (optimized with surfarray)."""
        # Clear screen
        self.screen.fill((0, 0, 0))

        if self.current_image is not None and NUMPY_AVAILABLE:
            # OPTIMIZED: Use numpy and surfarray instead of pixel-by-pixel
            # Simple nearest-neighbor scaling with numpy (no scipy needed)
            # Repeat each pixel to scale from 28x28 to ~112x112, then crop to 100x100
            scale_factor = 4  # 28 * 4 = 112
            scaled_image = np.repeat(np.repeat(self.current_image, scale_factor, axis=0), scale_factor, axis=1)
            # Crop to exact display size
            scaled_image = scaled_image[:self.display_height, :self.display_width]

            # Convert to 0-255 and create RGB array (pygame expects W, H, 3)
            rgb_array = (scaled_image * 255).astype(np.uint8)
            rgb_array = np.stack([rgb_array, rgb_array, rgb_array], axis=-1)

            # Transpose for pygame (expects width, height, 3)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))

            # Blit directly using surfarray
            pygame.surfarray.blit_array(self.screen, rgb_array)

        if self.render_mode == "human":
            pygame.display.flip()

    def get_observation(self):
        """Get current observation (flattened MNIST pixels)."""
        if self.observation_dirty:
            self._render_field()
            self._cache_observation()
            self.observation_dirty = False
        return self.cached_observation

    def _cache_observation(self):
        """Cache the MNIST image as flattened normalized pixels."""
        if self.current_image is not None and NUMPY_AVAILABLE:
            # Use the raw MNIST image (28x28) flattened
            # Already normalized 0-1 from torchvision
            self.cached_observation = self.current_image.flatten().tolist()
        else:
            # Fallback: capture from pygame screen
            if NUMPY_AVAILABLE:
                pixels = pygame.surfarray.array3d(self.screen)
                grayscale = np.mean(pixels, axis=2)
                normalized = grayscale / 255.0
                self.cached_observation = normalized.flatten().tolist()
            else:
                self.cached_observation = []
                for y in range(self.display_height):
                    for x in range(self.display_width):
                        pixel = self.screen.get_at((x, y))
                        gray = (pixel.r + pixel.g + pixel.b) / 3.0 / 255.0
                        self.cached_observation.append(gray)

    def step(self, digit_char):
        """
        Evaluate single digit output.

        Args:
            digit_char: single character (0-9)

        Returns:
            observation, reward, done, info
        """
        self.total_attempts += 1
        self.per_digit_attempts[self.current_digit] += 1

        # Normalize to string for comparison
        digit_char = str(digit_char).strip()
        if len(digit_char) > 0:
            digit_char = digit_char[0]
        else:
            digit_char = ""

        # Binary evaluation
        is_correct = (digit_char == self.current_digit)

        if is_correct:
            reward = cfg.MNIST_CORRECT_REWARD
            self.total_correct += 1
            self.per_digit_correct[self.current_digit] += 1
        else:
            reward = cfg.MNIST_WRONG_PENALTY

        # Move to next digit
        self.current_digit_idx += 1
        done = self.current_digit_idx >= cfg.MNIST_DIGITS_PER_EPISODE

        if not done:
            next_digit_idx = self.digit_order[self.current_digit_idx % 10]
            self._select_digit(next_digit_idx)

        info = {
            "is_correct": is_correct,
            "expected": self.current_digit if not done else str(self.digit_order[(self.current_digit_idx - 1) % 10]),
            "output": digit_char,
            "digit_idx": self.current_digit_idx - 1,
        }

        # Get new observation (next digit)
        observation = self.get_observation() if not done else self.cached_observation

        return observation, reward, done, info

    def get_accuracy_by_digit(self):
        """Get accuracy breakdown by digit."""
        result = {}
        for digit in self.digits:
            attempts = self.per_digit_attempts[digit]
            correct = self.per_digit_correct[digit]
            result[digit] = correct / max(1, attempts)
        return result

    def get_signals(self):
        """Get environment signals for protein cascade."""
        return {
            "accuracy": self.total_correct / max(1, self.total_attempts),
            "digit_progress": self.current_digit_idx / 10.0,
        }

    def render(self):
        """Force render current state."""
        self._render_field()
        if self.render_mode == "human":
            self.clock.tick(30)

    def close(self):
        """Cleanup pygame."""
        pygame.quit()


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def build_visual_protein_template():
    """
    Build GENREG protein template for visual learning.

    Returns list of proteins for visual perception.
    """
    from genreg_proteins import SensorProtein, TrendProtein, TrustModifierProtein

    proteins = []

    # Sensors for visual field signals
    proteins.append(SensorProtein("world_health"))
    proteins.append(SensorProtein("consecutive_correct"))
    proteins.append(SensorProtein("step_count"))
    proteins.append(SensorProtein("accuracy"))

    # Trend detection for performance
    health_trend = TrendProtein("health_trend")
    health_trend.bind_inputs(["world_health"])
    proteins.append(health_trend)

    # Trust modifier based on consecutive correct
    trust_visual = TrustModifierProtein("trust_visual")
    trust_visual.bind_inputs(["consecutive_correct"])
    trust_visual.params["gain"] = 1.0
    trust_visual.params["scale"] = 1.0
    proteins.append(trust_visual)

    return proteins


# ================================================================
# TESTING
# ================================================================
if __name__ == "__main__":
    print("Testing Visual Field Environment with Gradient Rewards...")
    print("=" * 60)

    # Test gradient calculation standalone
    print("\nGradient Calculation Tests:")
    print("-" * 40)
    test_cases = [
        ("sky", ["sky", "sea", "car"]),       # Exact match -> 100%
        ("ski", ["sky", "sea", "car"]),       # 2 of 3 chars match "sky" -> 67%
        ("sik", ["sky", "sea", "car"]),       # 1 of 3 chars match -> 33%
        ("xyz", ["sky", "sea", "car"]),       # No match -> 0%
        ("se", ["sky", "sea", "car"]),        # 2 of 3 match "sea" -> 67%
        ("car", ["sky", "sea", "car"]),       # Exact match -> 100%
        ("ca", ["sky", "sea", "car"]),        # 2 of 3 match "car" -> 67%
        ("xgpcclbeed", ["bed", "car"]),       # Garbage with "bed" inside -> LOW (30%)
        ("bedd", ["bed", "car"]),             # Close but extra char -> 75%
        ("be", ["bed", "car"]),               # Missing one char -> 67%
    ]

    for word, answers in test_cases:
        score, match, normalized = calculate_closeness(word, answers)
        reward = cfg.VISUAL_CLOSENESS_REWARD * normalized
        print(f"  '{word}' vs {answers}")
        print(f"    -> LCS={score}, closest='{match}', score={normalized:.2f}, reward=+{reward:.2f}")

    if not PYGAME_AVAILABLE:
        print("\nPygame not installed - cannot test visual env")
        exit(1)

    print("\n" + "=" * 60)
    print("Visual Environment Test:")
    print("-" * 40)

    env = VisualFieldEnv(render_mode="human")
    obs = env.reset()

    print(f"Observation size: {len(obs)}")
    print(f"Current sentence: {env.current_sentence}")
    print(f"Correct answers: {env.current_answers}")

    # Test gradient in action
    print("\nGradient Rewards Demo:")
    print("-" * 40)

    # Force a specific sentence for demo
    env.current_sentence = "the ____ is blue"
    env.current_answers = ["sky", "sea", "car"]
    env.observation_dirty = True

    test_words = [
        "xyz",  # No match -> full penalty
        "s",    # 1 char -> smaller penalty
        "sk",   # 2 chars -> smaller penalty
        "ski",  # 2 of 3 -> small penalty
        "sky",  # Exact match -> positive reward!
    ]

    for word in test_words:
        obs, reward, done, info = env.step(word)
        if info["is_correct"]:
            print(f"  '{word}' -> CORRECT! Reward: +{reward:.1f}")
            # Reset to same sentence for demo
            env.current_sentence = "the ____ is blue"
            env.current_answers = ["sky", "sea", "car"]
            env.observation_dirty = True
        else:
            closeness = info.get("closeness", {})
            print(f"  '{word}' -> closest='{closeness.get('closest_match')}' "
                  f"({closeness.get('normalized_score', 0):.0%}) "
                  f"Penalty: {reward:.2f}")
        env.render()
        pygame.time.wait(800)

    env.close()
    print("\n" + "=" * 60)
    print("Test complete! The gradient creates a smooth fitness landscape.")
    print("Evolution can now 'hill-climb' toward correct answers.")
