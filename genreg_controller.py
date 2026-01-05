# ================================================================
# GENREG v2.0 — Neural Controller (PyTorch GPU-Accelerated)
# ================================================================
# A small forward-pass-only neural network.
# - No backprop, no gradients
# - Weights evolve by mutation
# - GPU-accelerated with PyTorch
# ================================================================

import random
import math
import copy
import config as cfg

# Try to import PyTorch, fall back to CPU-only mode
try:
    import torch
    TORCH_AVAILABLE = True

    # Set default device based on config
    if cfg.DEVICE == "cuda":
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
            print(f"[CONTROLLER] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            DEVICE = torch.device('cpu')
            print("[CONTROLLER] CUDA requested but not available, using CPU")
    elif cfg.DEVICE == "cpu":
        DEVICE = torch.device('cpu')
        print("[CONTROLLER] Using CPU (forced by config)")
    else:  # "auto"
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
            print(f"[CONTROLLER] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            DEVICE = torch.device('cpu')
            print("[CONTROLLER] CUDA not available, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("[CONTROLLER] PyTorch not available, using pure Python (slow)")


# ================================================================
# GENREG Neural Controller (PyTorch Version)
# ================================================================
class GENREGController:
    def __init__(self, input_size, hidden_size=None, output_size=None, device=None):
        if hidden_size is None:
            hidden_size = cfg.HIDDEN_SIZE
        if output_size is None:
            output_size = cfg.OUTPUT_SIZE

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device if device is not None else DEVICE

        if TORCH_AVAILABLE and self.device is not None:
            self._init_torch()
        else:
            self._init_python()

    def _init_torch(self):
        """Initialize with PyTorch tensors on GPU/CPU."""
        w_min, w_max = cfg.WEIGHT_INIT_RANGE
        b_min, b_max = cfg.BIAS_INIT_RANGE

        # Create tensors on device
        self.w1 = torch.empty(self.hidden_size, self.input_size, device=self.device)
        self.w1.uniform_(w_min, w_max)

        self.b1 = torch.empty(self.hidden_size, device=self.device)
        self.b1.uniform_(b_min, b_max)

        self.w2 = torch.empty(self.output_size, self.hidden_size, device=self.device)
        self.w2.uniform_(w_min, w_max)

        self.b2 = torch.empty(self.output_size, device=self.device)
        self.b2.uniform_(b_min, b_max)

        self._use_torch = True

    def _init_python(self):
        """Fallback to pure Python lists."""
        w_min, w_max = cfg.WEIGHT_INIT_RANGE
        b_min, b_max = cfg.BIAS_INIT_RANGE

        self.w1 = [[random.uniform(w_min, w_max) for _ in range(self.input_size)]
                   for _ in range(self.hidden_size)]
        self.b1 = [random.uniform(b_min, b_max) for _ in range(self.hidden_size)]
        self.w2 = [[random.uniform(w_min, w_max) for _ in range(self.hidden_size)]
                   for _ in range(self.output_size)]
        self.b2 = [random.uniform(b_min, b_max) for _ in range(self.output_size)]

        self._use_torch = False

    # ------------------------------------------------------------
    def clone(self):
        """Deep copy controller and weights."""
        new = GENREGController.__new__(GENREGController)
        new.input_size = self.input_size
        new.hidden_size = self.hidden_size
        new.output_size = self.output_size
        new.device = self.device
        new._use_torch = self._use_torch

        if self._use_torch:
            new.w1 = self.w1.clone()
            new.b1 = self.b1.clone()
            new.w2 = self.w2.clone()
            new.b2 = self.b2.clone()
        else:
            new.w1 = copy.deepcopy(self.w1)
            new.b1 = copy.deepcopy(self.b1)
            new.w2 = copy.deepcopy(self.w2)
            new.b2 = copy.deepcopy(self.b2)

        return new

    # ------------------------------------------------------------
    def mutate(self, rate=None, scale=None):
        """Gaussian mutation across all weights."""
        if rate is None:
            rate = cfg.MUTATION_RATE
        if scale is None:
            scale = cfg.MUTATION_SCALE

        if self._use_torch:
            self._mutate_torch(rate, scale)
        else:
            self._mutate_python(rate, scale)

    def _mutate_torch(self, rate, scale):
        """GPU-accelerated mutation."""
        # Create mutation masks and noise
        mask1 = torch.rand_like(self.w1) < rate
        noise1 = torch.randn_like(self.w1) * scale
        self.w1 += mask1 * noise1

        mask_b1 = torch.rand_like(self.b1) < rate
        noise_b1 = torch.randn_like(self.b1) * scale
        self.b1 += mask_b1 * noise_b1

        mask2 = torch.rand_like(self.w2) < rate
        noise2 = torch.randn_like(self.w2) * scale
        self.w2 += mask2 * noise2

        mask_b2 = torch.rand_like(self.b2) < rate
        noise_b2 = torch.randn_like(self.b2) * scale
        self.b2 += mask_b2 * noise_b2

    def _mutate_python(self, rate, scale):
        """Pure Python mutation (slow fallback)."""
        def mutate_matrix(mat):
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if random.random() < rate:
                        mat[i][j] += random.gauss(0, scale)

        def mutate_vector(vec):
            for i in range(len(vec)):
                if random.random() < rate:
                    vec[i] += random.gauss(0, scale)

        mutate_matrix(self.w1)
        mutate_matrix(self.w2)
        mutate_vector(self.b1)
        mutate_vector(self.b2)

    # ------------------------------------------------------------
    def forward_text(self, inputs):
        """
        Text generation output (for Language Bootstrap).

        inputs: list[float] or torch.Tensor
        returns: tuple of (char_probs, countdown_value, string_length)
        """
        if self._use_torch:
            return self._forward_text_torch(inputs)
        else:
            return self._forward_text_python(inputs)

    def _forward_text_torch(self, inputs):
        """GPU-accelerated forward pass."""
        # Convert inputs to tensor if needed
        if not isinstance(inputs, torch.Tensor):
            x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        else:
            x = inputs.to(self.device)

        # Forward pass: hidden = tanh(W1 @ x + b1)
        hidden = torch.tanh(self.w1 @ x + self.b1)

        # Output = W2 @ hidden + b2
        outputs = self.w2 @ hidden + self.b2

        # Split outputs
        char_logits = outputs[:27]
        countdown_signal = outputs[27] if len(outputs) > 27 else torch.tensor(0.0, device=self.device)
        length_signal = outputs[28] if len(outputs) > 28 else torch.tensor(0.5, device=self.device)

        # Softmax for character probabilities
        char_probs = torch.softmax(char_logits, dim=0)

        # Sigmoid for countdown and length
        countdown_value = torch.sigmoid(countdown_signal)
        string_length = torch.sigmoid(length_signal)

        # Convert to Python for compatibility with existing code
        return char_probs.cpu().tolist(), countdown_value.item(), string_length.item()

    def _forward_text_python(self, inputs):
        """Pure Python forward pass (slow fallback)."""
        # Compute hidden layer
        hidden = []
        for i in range(self.hidden_size):
            s = self.b1[i]
            for j in range(self.input_size):
                s += self.w1[i][j] * inputs[j]
            hidden.append(math.tanh(s))

        # Compute outputs
        outputs = []
        for i in range(self.output_size):
            s = self.b2[i]
            for j in range(self.hidden_size):
                s += self.w2[i][j] * hidden[j]
            outputs.append(s)

        # Split outputs
        char_logits = outputs[:27] if len(outputs) >= 27 else outputs + [0.0] * (27 - len(outputs))
        countdown_signal = outputs[27] if len(outputs) > 27 else 0.0
        length_signal = outputs[28] if len(outputs) > 28 else 0.5

        # Softmax
        max_logit = max(char_logits)
        exp_logits = [math.exp(x - max_logit) for x in char_logits]
        sum_exp = sum(exp_logits)
        char_probs = [e / sum_exp for e in exp_logits]

        # Sigmoid
        countdown_value = 1.0 / (1.0 + math.exp(-countdown_signal))
        string_length = 1.0 / (1.0 + math.exp(-length_signal))

        return char_probs, countdown_value, string_length

    # ------------------------------------------------------------
    def sample_character(self, char_probs, temperature=1.0):
        """Sample a character from probability distribution."""
        if temperature != 1.0:
            scaled = [p ** (1.0 / temperature) for p in char_probs]
            total = sum(scaled)
            char_probs = [p / total for p in scaled]

        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(char_probs):
            cumsum += p
            if r <= cumsum:
                if i < 26:
                    return chr(ord('a') + i)
                else:
                    return ' '
        return ' '

    # ------------------------------------------------------------
    def generate_text(self, inputs, max_length=50, temperature=1.0, use_length_control=True):
        """Generate a string of text character by character."""
        if use_length_control:
            _, _, string_length = self.forward_text(inputs)
            actual_length = max(1, int(string_length * max_length))
        else:
            actual_length = max_length

        result = []
        for _ in range(actual_length):
            char_probs, _, _ = self.forward_text(inputs)
            char = self.sample_character(char_probs, temperature)
            result.append(char)

            char_idx = ord(char) - ord('a') if char != ' ' else 26
            char_signal = char_idx / 26.0
            if len(inputs) > 1:
                inputs = inputs[1:] + [char_signal]

        return ''.join(result)

    # ------------------------------------------------------------
    # VISUAL FIELD: Word generation from pixel input
    # ------------------------------------------------------------
    def forward_visual(self, visual_input):
        """
        Visual action selection from pixel input.

        visual_input: flattened pixel array (grayscale, normalized to [0,1])
        returns: (char_probs, ) - character probabilities
        """
        if self._use_torch:
            return self._forward_visual_torch(visual_input)
        else:
            return self._forward_visual_python(visual_input)

    def _forward_visual_torch(self, visual_input):
        """GPU-accelerated forward pass for visual input."""
        import torch
        if not isinstance(visual_input, torch.Tensor):
            x = torch.tensor(visual_input, dtype=torch.float32, device=self.device)
        else:
            x = visual_input.to(self.device)

        # Forward pass
        hidden = torch.tanh(self.w1 @ x + self.b1)
        outputs = self.w2 @ hidden + self.b2

        # All outputs are character logits
        char_logits = outputs[:27] if len(outputs) >= 27 else outputs
        char_probs = torch.softmax(char_logits, dim=0)

        return char_probs.cpu().tolist()

    def _forward_visual_python(self, visual_input):
        """Pure Python forward pass for visual input."""
        # Compute hidden layer
        hidden = []
        for i in range(self.hidden_size):
            s = self.b1[i]
            for j in range(min(self.input_size, len(visual_input))):
                s += self.w1[i][j] * visual_input[j]
            hidden.append(math.tanh(s))

        # Compute outputs
        outputs = []
        for i in range(min(self.output_size, 27)):
            s = self.b2[i]
            for j in range(self.hidden_size):
                s += self.w2[i][j] * hidden[j]
            outputs.append(s)

        # Softmax
        max_logit = max(outputs) if outputs else 0
        exp_logits = [math.exp(x - max_logit) for x in outputs]
        sum_exp = sum(exp_logits)
        char_probs = [e / sum_exp for e in exp_logits]

        return char_probs

    def generate_word(self, visual_input, max_length=None, temperature=1.0):
        """
        Generate a complete word from visual input.

        visual_input: flattened pixel array
        max_length: maximum word length (uses config if None)
        temperature: sampling temperature

        returns: generated word (string)
        """
        if max_length is None:
            max_length = cfg.MAX_WORD_LENGTH

        result = []
        current_input = list(visual_input)  # Copy to avoid modifying original

        for _ in range(max_length):
            char_probs = self.forward_visual(current_input)
            char = self.sample_character(char_probs, temperature)

            # Stop on space (word boundary)
            if char == ' ':
                break

            result.append(char)

            # Update input with character feedback (shift and add new signal)
            char_idx = ord(char) - ord('a')
            char_signal = char_idx / 26.0
            if len(current_input) > 1:
                current_input = current_input[1:] + [char_signal]

        return ''.join(result)

    def generate_char(self, visual_input, temperature=1.0):
        """
        Generate a single character from visual input.

        Used for alphabet recognition (Stage 1).
        visual_input: flattened pixel array
        temperature: sampling temperature

        returns: single character (a-z)
        """
        char_probs = self.forward_visual(visual_input)
        char = self.sample_character(char_probs, temperature)
        # Never return space for alphabet mode
        if char == ' ':
            # Pick the highest non-space probability
            probs_no_space = char_probs[:26]
            max_idx = probs_no_space.index(max(probs_no_space))
            char = chr(ord('a') + max_idx)
        return char

    def generate_digit(self, visual_input, temperature=1.0):
        """
        Generate a single digit from visual input.

        Used for MNIST digit recognition.
        visual_input: flattened pixel array (28x28 = 784)
        temperature: sampling temperature

        returns: single digit character ('0'-'9')
        """
        digit_probs = self.forward_visual(visual_input)
        # Take only first 10 outputs for digits
        digit_probs = digit_probs[:10]

        # Apply temperature
        if temperature != 1.0:
            # Convert to logits, apply temp, convert back
            import math
            logits = [math.log(max(p, 1e-10)) for p in digit_probs]
            logits = [l / temperature for l in logits]
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            digit_probs = [e / sum_exp for e in exp_logits]

        # Sample from distribution
        import random
        r = random.random()
        cumulative = 0.0
        selected_idx = 0
        for i, p in enumerate(digit_probs):
            cumulative += p
            if r <= cumulative:
                selected_idx = i
                break

        return str(selected_idx)

    def generate_category(self, visual_input, num_categories=None, temperature=1.0):
        """
        Generate a category index from visual input.

        Used for Caltech-101 category recognition.
        visual_input: flattened pixel array
        num_categories: number of output categories (uses output_size if None)
        temperature: sampling temperature

        returns: category index (integer)
        """
        if num_categories is None:
            num_categories = self.output_size

        category_probs = self.forward_visual(visual_input)
        # Take only first num_categories outputs
        category_probs = category_probs[:num_categories]

        # Apply temperature
        if temperature != 1.0:
            import math
            logits = [math.log(max(p, 1e-10)) for p in category_probs]
            logits = [l / temperature for l in logits]
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            category_probs = [e / sum_exp for e in exp_logits]

        # Sample from distribution
        import random
        r = random.random()
        cumulative = 0.0
        selected_idx = 0
        for i, p in enumerate(category_probs):
            cumulative += p
            if r <= cumulative:
                selected_idx = i
                break

        return selected_idx

    # ------------------------------------------------------------
    # BATCHED OPERATIONS - Process all genomes at once
    # ------------------------------------------------------------
    def forward_text_batched(self, inputs_batch):
        """
        Batched forward pass for multiple inputs at once.

        inputs_batch: tensor of shape (batch_size, input_size)
        returns: (char_probs_batch, countdown_batch, length_batch)
        """
        if not self._use_torch:
            raise RuntimeError("Batched operations require PyTorch")

        # inputs_batch: (B, input_size)
        # w1: (hidden_size, input_size)
        # hidden = tanh(inputs @ w1.T + b1)

        hidden = torch.tanh(inputs_batch @ self.w1.T + self.b1)  # (B, hidden_size)
        outputs = hidden @ self.w2.T + self.b2  # (B, output_size)

        char_logits = outputs[:, :27]
        countdown_signal = outputs[:, 27] if outputs.shape[1] > 27 else torch.zeros(outputs.shape[0], device=self.device)
        length_signal = outputs[:, 28] if outputs.shape[1] > 28 else torch.full((outputs.shape[0],), 0.5, device=self.device)

        char_probs = torch.softmax(char_logits, dim=1)
        countdown_value = torch.sigmoid(countdown_signal)
        string_length = torch.sigmoid(length_signal)

        return char_probs, countdown_value, string_length

    # ------------------------------------------------------------
    def forward(self, inputs):
        """Discrete action selection (for Snake)."""
        if self._use_torch:
            if not isinstance(inputs, torch.Tensor):
                x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            else:
                x = inputs.to(self.device)
            hidden = torch.tanh(self.w1 @ x + self.b1)
            outputs = self.w2 @ hidden + self.b2
            return torch.argmax(outputs).item()
        else:
            hidden = []
            for i in range(self.hidden_size):
                s = self.b1[i]
                for j in range(self.input_size):
                    s += self.w1[i][j] * inputs[j]
                hidden.append(math.tanh(s))
            outputs = []
            for i in range(self.output_size):
                s = self.b2[i]
                for j in range(self.hidden_size):
                    s += self.w2[i][j] * hidden[j]
                outputs.append(s)
            return max(range(self.output_size), key=lambda i: outputs[i])

    # ------------------------------------------------------------
    def forward_continuous(self, inputs):
        """Continuous action output (for Walker2D)."""
        if self._use_torch:
            if not isinstance(inputs, torch.Tensor):
                x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            else:
                x = inputs.to(self.device)
            hidden = torch.tanh(self.w1 @ x + self.b1)
            outputs = self.w2 @ hidden + self.b2
            return torch.tanh(outputs).cpu().tolist()
        else:
            hidden = []
            for i in range(self.hidden_size):
                s = self.b1[i]
                for j in range(self.input_size):
                    s += self.w1[i][j] * inputs[j]
                hidden.append(math.tanh(s))
            outputs = []
            for i in range(self.output_size):
                s = self.b2[i]
                for j in range(self.hidden_size):
                    s += self.w2[i][j] * hidden[j]
                outputs.append(math.tanh(s))
            return outputs


# ================================================================
# BATCHED CONTROLLER - Runs ALL genomes in a single GPU call
# ================================================================
class BatchedController:
    """
    Runs forward passes for ALL genomes in a single batched GPU operation.
    Much faster than individual forward passes.
    """

    def __init__(self, controllers, device=None):
        """
        Stack all controller weights into batched tensors.

        controllers: list of GENREGController objects
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("BatchedController requires PyTorch")

        self.device = device if device is not None else DEVICE
        self.n_controllers = len(controllers)

        # Get dimensions from first controller
        c0 = controllers[0]
        self.input_size = c0.input_size
        self.hidden_size = c0.hidden_size
        self.output_size = c0.output_size

        # Stack all weights: (n_controllers, hidden_size, input_size)
        self.w1 = torch.stack([c.w1 for c in controllers]).to(self.device)
        self.b1 = torch.stack([c.b1 for c in controllers]).to(self.device)
        self.w2 = torch.stack([c.w2 for c in controllers]).to(self.device)
        self.b2 = torch.stack([c.b2 for c in controllers]).to(self.device)

    def forward_text_all(self, inputs_list):
        """
        Run forward pass for ALL controllers with their respective inputs.

        inputs_list: list of input vectors, one per controller
                    OR tensor of shape (n_controllers, input_size)

        returns: (char_probs, countdowns, lengths) - all as tensors
        """
        if isinstance(inputs_list, list):
            inputs = torch.tensor(inputs_list, dtype=torch.float32, device=self.device)
        else:
            inputs = inputs_list.to(self.device)

        # inputs: (N, input_size)
        # w1: (N, hidden_size, input_size)
        # We need: hidden[i] = tanh(w1[i] @ inputs[i] + b1[i])

        # Use einsum for batched matrix-vector multiplication
        # 'nhi,ni->nh' means: for each n, multiply (h,i) matrix with (i,) vector -> (h,) result
        hidden = torch.tanh(torch.einsum('nhi,ni->nh', self.w1, inputs) + self.b1)

        # outputs[i] = w2[i] @ hidden[i] + b2[i]
        outputs = torch.einsum('noh,nh->no', self.w2, hidden) + self.b2

        # Split outputs
        char_logits = outputs[:, :27]
        countdown_signal = outputs[:, 27] if outputs.shape[1] > 27 else torch.zeros(self.n_controllers, device=self.device)
        length_signal = outputs[:, 28] if outputs.shape[1] > 28 else torch.full((self.n_controllers,), 0.5, device=self.device)

        char_probs = torch.softmax(char_logits, dim=1)
        countdown_value = torch.sigmoid(countdown_signal)
        string_length = torch.sigmoid(length_signal)

        return char_probs, countdown_value, string_length

    def sample_characters_all(self, char_probs, temperature=1.0):
        """
        Sample characters for all controllers at once.

        char_probs: tensor of shape (n_controllers, 27)
        returns: list of characters
        """
        if temperature != 1.0:
            char_probs = char_probs ** (1.0 / temperature)
            char_probs = char_probs / char_probs.sum(dim=1, keepdim=True)

        # Sample using multinomial
        indices = torch.multinomial(char_probs, 1).squeeze(1)  # (N,)

        # Convert to characters
        chars = []
        for idx in indices.cpu().tolist():
            if idx < 26:
                chars.append(chr(ord('a') + idx))
            else:
                chars.append(' ')

        return chars

    def sync_from_controllers(self, controllers):
        """Update batched weights from individual controllers (after mutation)."""
        self.w1 = torch.stack([c.w1 for c in controllers]).to(self.device)
        self.b1 = torch.stack([c.b1 for c in controllers]).to(self.device)
        self.w2 = torch.stack([c.w2 for c in controllers]).to(self.device)
        self.b2 = torch.stack([c.b2 for c in controllers]).to(self.device)

    def forward_visual_all(self, obs):
        """
        Run visual forward pass for ALL controllers with the SAME observation.

        obs: single observation vector (all genomes see the same image)
        returns: char_probs tensor of shape (n_controllers, 26)
        """
        # Convert observation to tensor and expand to batch
        if isinstance(obs, list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)

        # Expand obs to (n_controllers, input_size) - all same input
        inputs = obs_tensor.unsqueeze(0).expand(self.n_controllers, -1)

        # Batched forward pass
        # hidden[i] = tanh(w1[i] @ obs + b1[i])
        hidden = torch.tanh(torch.einsum('nhi,ni->nh', self.w1, inputs) + self.b1)
        outputs = torch.einsum('noh,nh->no', self.w2, hidden) + self.b2

        # All outputs are character logits (26 letters for alphabet mode)
        char_logits = outputs[:, :26]
        char_probs = torch.softmax(char_logits, dim=1)

        return char_probs

    def forward_visual_all_27(self, inputs):
        """
        Run visual forward pass for ALL controllers with per-genome inputs.

        inputs: tensor of shape (n_controllers, input_size)
        returns: char_probs tensor of shape (n_controllers, 27) including space
        """
        # Batched forward pass
        hidden = torch.tanh(torch.einsum('nhi,ni->nh', self.w1, inputs) + self.b1)
        outputs = torch.einsum('noh,nh->no', self.w2, hidden) + self.b2

        # 27 character logits (a-z + space)
        char_logits = outputs[:, :27]
        char_probs = torch.softmax(char_logits, dim=1)

        return char_probs

    def generate_word_all(self, obs, max_length=None, temperature=1.0):
        """
        Generate complete words for ALL controllers from the same observation.

        All genomes start with the same visual input and generate words in parallel.
        Each genome generates characters until it outputs a space or hits max_length.

        obs: single observation vector (all genomes see the same image)
        max_length: maximum word length (uses config if None)
        temperature: sampling temperature

        returns: list of generated words (one per controller)
        """
        if max_length is None:
            max_length = cfg.MAX_WORD_LENGTH

        # Convert observation to tensor
        if isinstance(obs, list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)

        # Each genome gets its own copy of the input (will diverge as chars are generated)
        inputs = obs_tensor.unsqueeze(0).expand(self.n_controllers, -1).clone()

        # Track results and which genomes are done
        results = [[] for _ in range(self.n_controllers)]
        done = torch.zeros(self.n_controllers, dtype=torch.bool, device=self.device)

        for _ in range(max_length):
            if done.all():
                break

            # Forward pass for all genomes
            char_probs = self.forward_visual_all_27(inputs)

            # Apply temperature
            if temperature != 1.0:
                char_probs = char_probs ** (1.0 / temperature)
                char_probs = char_probs / char_probs.sum(dim=1, keepdim=True)

            # Sample characters
            sampled_indices = torch.multinomial(char_probs, 1).squeeze(1)  # (N,)

            # Process each genome
            for i in range(self.n_controllers):
                if done[i]:
                    continue

                idx = sampled_indices[i].item()
                if idx < 26:
                    char = chr(ord('a') + idx)
                    results[i].append(char)
                else:
                    # Space = word boundary, mark as done
                    done[i] = True
                    continue

            # Update inputs with character feedback for non-done genomes
            # Shift input and add new character signal
            char_signals = sampled_indices.float() / 26.0
            char_signals = char_signals.clamp(0, 1)  # Normalize

            # Shift: drop first element, append char signal
            inputs = torch.cat([inputs[:, 1:], char_signals.unsqueeze(1)], dim=1)

        return [''.join(r) for r in results]

    def generate_char_all(self, obs, temperature=1.0):
        """
        Generate a single character for ALL controllers from the same observation.

        obs: single observation vector
        temperature: sampling temperature
        returns: list of characters (one per controller)
        """
        char_probs = self.forward_visual_all(obs)

        if temperature != 1.0:
            char_probs = char_probs ** (1.0 / temperature)
            char_probs = char_probs / char_probs.sum(dim=1, keepdim=True)

        # Sample characters using multinomial
        indices = torch.multinomial(char_probs, 1).squeeze(1)  # (N,)

        # Convert to characters (a-z only, no space for alphabet mode)
        chars = []
        for idx in indices.cpu().tolist():
            if idx < 26:
                chars.append(chr(ord('a') + idx))
            else:
                # Fallback: pick highest prob letter
                chars.append('a')  # Should not happen with 26 outputs

        return chars

    def forward_visual_all_digits(self, obs):
        """
        Run visual forward pass for ALL controllers for MNIST digit recognition.

        obs: single observation vector (all genomes see the same image)
        returns: digit_probs tensor of shape (n_controllers, 10)
        """
        # Convert observation to tensor and expand to batch
        if isinstance(obs, list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)

        # Expand obs to (n_controllers, input_size) - all same input
        inputs = obs_tensor.unsqueeze(0).expand(self.n_controllers, -1)

        # Batched forward pass
        hidden = torch.tanh(torch.einsum('nhi,ni->nh', self.w1, inputs) + self.b1)
        outputs = torch.einsum('noh,nh->no', self.w2, hidden) + self.b2

        # All outputs are digit logits (10 digits for MNIST mode)
        digit_logits = outputs[:, :10]
        digit_probs = torch.softmax(digit_logits, dim=1)

        return digit_probs

    def generate_digit_all(self, obs, temperature=1.0):
        """
        Generate a single digit for ALL controllers from the same observation.

        obs: single observation vector (MNIST 28x28 = 784)
        temperature: sampling temperature
        returns: list of digit strings (one per controller)
        """
        digit_probs = self.forward_visual_all_digits(obs)

        if temperature != 1.0:
            digit_probs = digit_probs ** (1.0 / temperature)
            digit_probs = digit_probs / digit_probs.sum(dim=1, keepdim=True)

        # Sample digits using multinomial
        indices = torch.multinomial(digit_probs, 1).squeeze(1)  # (N,)

        # Convert to digit strings
        digits = [str(idx) for idx in indices.cpu().tolist()]

        return digits

    def forward_visual_all_categories(self, obs, num_categories):
        """
        Run visual forward pass for ALL controllers for category recognition.

        obs: single observation vector (all genomes see the same image)
        num_categories: number of output categories
        returns: category_probs tensor of shape (n_controllers, num_categories)
        """
        # Convert observation to tensor and expand to batch
        if isinstance(obs, list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)

        # Expand obs to (n_controllers, input_size) - all same input
        inputs = obs_tensor.unsqueeze(0).expand(self.n_controllers, -1)

        # Batched forward pass
        hidden = torch.tanh(torch.einsum('nhi,ni->nh', self.w1, inputs) + self.b1)
        outputs = torch.einsum('noh,nh->no', self.w2, hidden) + self.b2

        # Take first num_categories outputs as category logits
        category_logits = outputs[:, :num_categories]
        category_probs = torch.softmax(category_logits, dim=1)

        return category_probs

    def generate_category_all(self, obs, temperature=1.0):
        """
        Generate a single category index for ALL controllers from the same observation.

        obs: single observation vector
        temperature: sampling temperature
        returns: list of category indices (integers, one per controller)
        """
        # Use the output_size as number of categories
        category_probs = self.forward_visual_all_categories(obs, self.output_size)

        if temperature != 1.0:
            category_probs = category_probs ** (1.0 / temperature)
            category_probs = category_probs / category_probs.sum(dim=1, keepdim=True)

        # Sample categories using multinomial
        indices = torch.multinomial(category_probs, 1).squeeze(1)  # (N,)

        # Convert to list of integers
        return indices.cpu().tolist()

    def generate_text_all(self, inputs_list, max_length=50, temperature=1.0, indices=None):
        """
        Generate text for selected controllers in parallel.

        inputs_list: list of input vectors
        max_length: maximum characters to generate
        temperature: sampling temperature
        indices: optional list of controller indices to use (if None, assumes sequential 0..len(inputs))

        returns: list of generated strings
        """
        n = len(inputs_list)
        if n == 0:
            return []

        # If indices provided, create a sub-batched forward function
        if indices is not None:
            w1 = self.w1[indices]
            b1 = self.b1[indices]
            w2 = self.w2[indices]
            b2 = self.b2[indices]
        else:
            w1 = self.w1[:n]
            b1 = self.b1[:n]
            w2 = self.w2[:n]
            b2 = self.b2[:n]

        # Convert inputs to tensor
        if isinstance(inputs_list, list):
            inputs = torch.tensor(inputs_list, dtype=torch.float32, device=self.device)
        else:
            inputs = inputs_list.to(self.device)

        # Forward pass for subset
        hidden = torch.tanh(torch.einsum('nhi,ni->nh', w1, inputs) + b1)
        outputs = torch.einsum('noh,nh->no', w2, hidden) + b2

        # Get string lengths
        length_signal = outputs[:, 28] if outputs.shape[1] > 28 else torch.full((n,), 0.5, device=self.device)
        lengths = torch.sigmoid(length_signal)
        actual_lengths = (lengths * max_length).clamp(min=1).int()

        # Generate characters in parallel
        results = [[] for _ in range(n)]
        max_len = actual_lengths.max().item()

        for char_idx in range(max_len):
            # Forward pass
            hidden = torch.tanh(torch.einsum('nhi,ni->nh', w1, inputs) + b1)
            outputs = torch.einsum('noh,nh->no', w2, hidden) + b2

            char_logits = outputs[:, :27]
            char_probs = torch.softmax(char_logits, dim=1)

            # Sample characters
            if temperature != 1.0:
                char_probs = char_probs ** (1.0 / temperature)
                char_probs = char_probs / char_probs.sum(dim=1, keepdim=True)

            sampled_indices = torch.multinomial(char_probs, 1).squeeze(1)

            chars = []
            for idx in sampled_indices.cpu().tolist():
                if idx < 26:
                    chars.append(chr(ord('a') + idx))
                else:
                    chars.append(' ')

            # Only add char if within this genome's length
            for i in range(n):
                if char_idx < actual_lengths[i].item():
                    results[i].append(chars[i])

            # Update inputs with character feedback
            char_indices = torch.tensor(
                [ord(c) - ord('a') if c != ' ' else 26 for c in chars],
                dtype=torch.float32, device=self.device
            ) / 26.0

            # Shift inputs and add new signal
            inputs = torch.cat([inputs[:, 1:], char_indices.unsqueeze(1)], dim=1)

        return [''.join(r) for r in results]

    def select_word_all(self, obs, vocabulary, temperature=1.0):
        """
        Select a word from vocabulary for ALL controllers from the same observation.

        Instead of generating characters, the network outputs scores over vocabulary.
        This is vocabulary-constrained decoding - massively reduces search space.

        obs: single observation vector (all genomes see same image)
        vocabulary: list of valid words to choose from
        temperature: sampling temperature

        returns: list of selected words (one per controller)
        """
        vocab_size = len(vocabulary)

        # Convert observation to tensor and expand to batch
        if isinstance(obs, list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)

        # Expand obs to (n_controllers, input_size) - all same input
        inputs = obs_tensor.unsqueeze(0).expand(self.n_controllers, -1)

        # Batched forward pass
        hidden = torch.tanh(torch.einsum('nhi,ni->nh', self.w1, inputs) + self.b1)
        outputs = torch.einsum('noh,nh->no', self.w2, hidden) + self.b2

        # Use outputs as vocab scores - take first vocab_size outputs
        # If output_size < vocab_size, we repeat/tile to fill
        # If output_size > vocab_size, we slice
        if self.output_size >= vocab_size:
            vocab_logits = outputs[:, :vocab_size]
        else:
            # Tile outputs to match vocab size
            repeats = (vocab_size // self.output_size) + 1
            tiled = outputs.repeat(1, repeats)[:, :vocab_size]
            vocab_logits = tiled

        # Apply softmax to get probabilities
        vocab_probs = torch.softmax(vocab_logits, dim=1)

        # Apply temperature
        if temperature != 1.0:
            vocab_probs = vocab_probs ** (1.0 / temperature)
            vocab_probs = vocab_probs / vocab_probs.sum(dim=1, keepdim=True)

        # Sample word indices
        word_indices = torch.multinomial(vocab_probs, 1).squeeze(1)  # (N,)

        # Convert to words
        words = [vocabulary[idx] for idx in word_indices.cpu().tolist()]

        return words
