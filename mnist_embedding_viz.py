# ================================================================
# GENREG MNIST Embedding Visualization (Standalone)
# ================================================================
# Self-contained script to visualize hidden layer activations.
# No external dependencies on config.py or genreg_controller.py.
#
# Usage:
#   python mnist_embedding_viz.py
#
# Output:
#   - PNG visualization of hidden layer embeddings colored by digit
#   - Saved to best_genomes/ folder
# ================================================================

import os
import sys
import pickle
import time
from datetime import datetime

import numpy as np


# ================================================================
# SIMPLE NETWORK WEIGHTS CONTAINER
# ================================================================
class NetworkWeights:
    """Simple container for neural network weights (numpy-based)."""

    def __init__(self):
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def load_weights(self, ctrl_data):
        """Load weights from checkpoint data as numpy arrays."""
        self.w1 = np.array(ctrl_data['w1'])
        self.b1 = np.array(ctrl_data['b1'])
        self.w2 = np.array(ctrl_data['w2'])
        self.b2 = np.array(ctrl_data['b2'])


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
    print("SELECT MNIST GENOME FOR EMBEDDING VISUALIZATION")
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

    ctrl_data = data['controller']
    input_size = ctrl_data['input_size']
    hidden_size = ctrl_data['hidden_size']
    output_size = ctrl_data['output_size']

    print(f"  Network: {input_size} -> {hidden_size} -> {output_size}")
    print(f"  Hidden layer dimension: {hidden_size}")

    # Create weights container and load weights
    network = NetworkWeights()
    network.load_weights(ctrl_data)

    trust = data.get('genome', {}).get('trust', 0)
    print(f"  Trust: {trust:.2f}")

    return network, data, hidden_size


# ================================================================
# MNIST DATASET LOADING
# ================================================================
def load_mnist_test_set():
    """Load the official MNIST 10K test set."""
    print("\nLoading MNIST test set...")

    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("ERROR: torchvision required. Install with: pip install torchvision")
        sys.exit(1)

    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    print(f"  Loaded {len(test_dataset)} test images")
    return test_dataset


# ================================================================
# HIDDEN LAYER EXTRACTION
# ================================================================
def extract_hidden_activations(network, test_dataset):
    """
    Run all test images through the network and capture hidden layer activations.

    Returns:
        embeddings: numpy array of shape (10000, hidden_size)
        labels: numpy array of shape (10000,)
        predictions: numpy array of shape (10000,)
    """
    print("\n" + "=" * 60)
    print("EXTRACTING HIDDEN LAYER ACTIVATIONS")
    print("=" * 60)

    embeddings = []
    labels = []
    predictions = []

    total = len(test_dataset)
    start_time = time.perf_counter()

    print(f"\nProcessing {total} images...")

    for idx, (image, label) in enumerate(test_dataset):
        # Flatten image to 784 pixels
        obs = image.view(-1).numpy()

        # Forward pass to hidden layer: hidden = tanh(w1 @ obs + b1)
        hidden = np.tanh(network.w1 @ obs + network.b1)

        # Output layer for prediction
        output = network.w2 @ hidden + network.b2
        predicted = int(np.argmax(output[:10]))

        embeddings.append(hidden)
        labels.append(label)
        predictions.append(predicted)

        if (idx + 1) % 2000 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed
            print(f"  Progress: {idx + 1}/{total} ({rate:.0f} images/sec)")

    elapsed = time.perf_counter() - start_time
    print(f"\nExtraction complete in {elapsed:.2f} seconds")

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Calculate accuracy
    correct = np.sum(predictions == labels)
    accuracy = correct / total * 100
    print(f"Accuracy on test set: {accuracy:.2f}% ({correct}/{total})")

    return embeddings, labels, predictions


# ================================================================
# DIMENSIONALITY REDUCTION
# ================================================================
def reduce_dimensions(embeddings, method='tsne'):
    """
    Reduce embeddings to 2D using t-SNE or UMAP.

    Args:
        embeddings: (N, hidden_size) array
        method: 'tsne' or 'umap'

    Returns:
        (N, 2) array of 2D coordinates
    """
    print(f"\n" + "=" * 60)
    print(f"DIMENSIONALITY REDUCTION ({method.upper()})")
    print("=" * 60)

    n_samples, hidden_dim = embeddings.shape
    print(f"\nReducing {n_samples} samples from {hidden_dim}D to 2D...")

    start_time = time.perf_counter()

    if method == 'tsne':
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
            sys.exit(1)

        print("  Using t-SNE (this may take a few minutes)...")
        print("  Parameters: perplexity=30, max_iter=1000, learning_rate='auto'")

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=1000,
            learning_rate='auto',
            init='pca',
            random_state=42,
            verbose=1
        )
        coords_2d = tsne.fit_transform(embeddings)

    elif method == 'umap':
        try:
            import umap
        except ImportError:
            print("ERROR: umap-learn required. Install with: pip install umap-learn")
            print("Falling back to t-SNE...")
            return reduce_dimensions(embeddings, method='tsne')

        print("  Using UMAP...")
        print("  Parameters: n_neighbors=15, min_dist=0.1")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            verbose=True
        )
        coords_2d = reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.perf_counter() - start_time
    print(f"\nReduction complete in {elapsed:.2f} seconds")

    return coords_2d


# ================================================================
# VISUALIZATION
# ================================================================
def create_visualization(coords_2d, labels, predictions, genome_info, method, output_dir):
    """
    Create and save visualization of embeddings.
    """
    print(f"\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        sys.exit(1)

    # Color palette for 10 digits
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # ---- Plot 1: Colored by TRUE label ----
    ax1 = axes[0]
    for digit in range(10):
        mask = labels == digit
        ax1.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[digit]],
            label=str(digit),
            alpha=0.6,
            s=5
        )

    ax1.set_title(f'Hidden Layer Embeddings - Colored by TRUE Label\n'
                  f'Genome: {genome_info["filename"]} (Gen {genome_info["generation"]})',
                  fontsize=12)
    ax1.set_xlabel(f'{method.upper()} Dimension 1')
    ax1.set_ylabel(f'{method.upper()} Dimension 2')
    ax1.legend(title='Digit', loc='upper right', markerscale=3)

    # ---- Plot 2: Colored by PREDICTED label (shows errors) ----
    ax2 = axes[1]

    # First plot correct predictions
    correct_mask = predictions == labels
    for digit in range(10):
        mask = (predictions == digit) & correct_mask
        ax2.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[digit]],
            alpha=0.5,
            s=5
        )

    # Then plot errors as X markers
    error_mask = predictions != labels
    ax2.scatter(
        coords_2d[error_mask, 0],
        coords_2d[error_mask, 1],
        c='red',
        marker='x',
        s=20,
        alpha=0.8,
        label=f'Errors ({np.sum(error_mask)})'
    )

    accuracy = np.sum(correct_mask) / len(labels) * 100
    ax2.set_title(f'Hidden Layer Embeddings - Colored by PREDICTION\n'
                  f'Accuracy: {accuracy:.2f}% (Red X = errors)',
                  fontsize=12)
    ax2.set_xlabel(f'{method.upper()} Dimension 1')
    ax2.set_ylabel(f'{method.upper()} Dimension 2')

    # Create legend for predictions
    patches = [mpatches.Patch(color=colors[i], label=str(i)) for i in range(10)]
    patches.append(mpatches.Patch(color='red', label=f'Errors'))
    ax2.legend(handles=patches, title='Predicted', loc='upper right', markerscale=3)

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen = genome_info['generation']
    output_filename = f"mnist_embeddings_gen{gen}_{method}_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Also create a larger single plot version
    fig2, ax = plt.subplots(figsize=(14, 12))

    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[digit]],
            label=f'{digit} (n={np.sum(mask)})',
            alpha=0.6,
            s=8
        )

    ax.set_title(f'GENREG MNIST Hidden Layer Embeddings ({method.upper()})\n'
                 f'Genome: Gen {genome_info["generation"]} | '
                 f'Hidden Dim: {coords_2d.shape[0]} samples | '
                 f'Accuracy: {accuracy:.2f}%',
                 fontsize=14)
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.legend(title='Digit', loc='upper right', markerscale=2, fontsize=10)

    output_filename2 = f"mnist_embeddings_gen{gen}_{method}_{timestamp}_large.png"
    output_path2 = os.path.join(output_dir, output_filename2)
    plt.savefig(output_path2, dpi=200, bbox_inches='tight')
    print(f"Large visualization saved to: {output_path2}")

    plt.close('all')

    return output_path, output_path2


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "=" * 60)
    print("GENREG MNIST EMBEDDING VISUALIZATION")
    print("Hidden Layer Activation Analysis")
    print("=" * 60)

    # Select genome
    genome_info = select_genome()

    # Load genome
    network, data, hidden_size = load_genome(genome_info)

    # Load MNIST test set
    test_dataset = load_mnist_test_set()

    # Extract hidden layer activations
    embeddings, labels, predictions = extract_hidden_activations(network, test_dataset)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"  - {embeddings.shape[0]} samples")
    print(f"  - {embeddings.shape[1]} dimensions (hidden layer size)")

    # Ask for reduction method
    print("\n" + "-" * 40)
    print("Select dimensionality reduction method:")
    print("  [1] t-SNE (slower but often better clusters)")
    print("  [2] UMAP (faster, preserves global structure)")
    print("  [3] Both")
    choice = input("\nYour choice [1]: ").strip()

    output_dir = "best_genomes"

    if choice == "2":
        methods = ['umap']
    elif choice == "3":
        methods = ['tsne', 'umap']
    else:
        methods = ['tsne']

    for method in methods:
        # Reduce dimensions
        coords_2d = reduce_dimensions(embeddings, method=method)

        # Create and save visualization
        create_visualization(coords_2d, labels, predictions, genome_info, method, output_dir)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
