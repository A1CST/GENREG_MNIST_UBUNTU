# ================================================================
# GENREG Checkpoint System
# ================================================================
# Saves and loads complete population state for continued training
# Supports both PyTorch tensors and Python lists
# Supports visual field and language bootstrap modes
# ================================================================

import pickle
import os
from pathlib import Path

# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def save_checkpoint(population, generation, template_proteins, checkpoint_dir="checkpoints", phase_manager_state=None):
    """
    Save a complete checkpoint of the population state.

    Args:
        population: GENREGPopulation instance
        generation: Current generation number
        template_proteins: List of template proteins (for reference)
        checkpoint_dir: Directory to save checkpoints in
        phase_manager_state: Optional dict from PhaseManager.get_state() for Phase 2 support
    """
    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint data
    checkpoint_data = {
        "generation": generation,
        "population_size": population.size,
        "input_size": population.genomes[0].controller.input_size if population.genomes else None,
        "hidden_size": population.genomes[0].controller.hidden_size if population.genomes else None,
        "output_size": population.genomes[0].controller.output_size if population.genomes else None,
        "template_proteins": template_proteins,
        "genomes": [],
        "phase_manager_state": phase_manager_state,  # Phase 2 support
    }
    
    # Save each genome's complete state
    for genome in population.genomes:
        # Convert tensors to CPU lists for pickle compatibility
        controller = genome.controller
        if TORCH_AVAILABLE and hasattr(controller, '_use_torch') and controller._use_torch:
            w1 = controller.w1.cpu().tolist()
            b1 = controller.b1.cpu().tolist()
            w2 = controller.w2.cpu().tolist()
            b2 = controller.b2.cpu().tolist()
        else:
            w1 = controller.w1
            b1 = controller.b1
            w2 = controller.w2
            b2 = controller.b2

        genome_data = {
            "id": genome.id,
            "trust": genome.trust,
            "food_eaten": getattr(genome, 'food_eaten', 0),
            "step_count": getattr(genome, 'step_count', 0),
            "proteins": genome.proteins,  # Full protein objects with params and state
            "controller": {
                "input_size": controller.input_size,
                "hidden_size": controller.hidden_size,
                "output_size": controller.output_size,
                "w1": w1,
                "b1": b1,
                "w2": w2,
                "b2": b2,
            },
            # Visual field state
            "visual_state": {
                "correct_predictions": getattr(genome, 'correct_predictions', 0),
                "total_predictions": getattr(genome, 'total_predictions', 0),
                "consecutive_correct": getattr(genome, 'consecutive_correct', 0),
            },
            # Stagnation tracking
            "stagnation_state": {
                "last_wrong_word": getattr(genome, 'last_wrong_word', ''),
                "stagnation_count": getattr(genome, 'stagnation_count', 0),
            }
        }
        checkpoint_data["genomes"].append(genome_data)
    
    # Save active genome index
    checkpoint_data["active_index"] = population.active
    
    # Create checkpoint filename
    checkpoint_filename = f"checkpoint_gen_{generation:05d}.pkl"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Save checkpoint
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"[CHECKPOINT] Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, template_proteins=None):
    """
    Load a checkpoint and reconstruct the population.

    Args:
        checkpoint_path: Path to checkpoint file
        template_proteins: Optional template proteins (will use from checkpoint if None)

    Returns:
        tuple: (population, generation, template_proteins, phase_manager_state)
               phase_manager_state may be None for older checkpoints
    """
    from genreg_genome import GENREGPopulation, GENREGGenome
    from genreg_controller import GENREGController

    # Check file exists and has content
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    file_size = os.path.getsize(checkpoint_path)
    if file_size == 0:
        raise ValueError(f"Checkpoint file is empty (0 bytes): {checkpoint_path}")

    # Load checkpoint data with better error handling
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
    except EOFError as e:
        # File was truncated or corrupted
        raise ValueError(
            f"Checkpoint file is corrupted or incomplete ({file_size} bytes): {checkpoint_path}\n"
            f"This usually means the save was interrupted. Try an older checkpoint."
        ) from e
    except pickle.UnpicklingError as e:
        raise ValueError(
            f"Checkpoint file has invalid pickle data: {checkpoint_path}\n"
            f"Error: {e}"
        ) from e

    generation = checkpoint_data["generation"]
    population_size = checkpoint_data["population_size"]
    phase_manager_state = checkpoint_data.get("phase_manager_state", None)  # Phase 2 support

    # Get or use provided template proteins
    if template_proteins is None:
        template_proteins = checkpoint_data["template_proteins"]
    
    # Extract network architecture
    input_size = checkpoint_data["input_size"]
    hidden_size = checkpoint_data["hidden_size"]
    output_size = checkpoint_data["output_size"]
    
    # Reconstruct genomes
    genomes = []
    for genome_data in checkpoint_data["genomes"]:
        # Reconstruct controller
        controller = GENREGController(input_size, hidden_size, output_size)

        # Load weights, converting to tensors if using PyTorch
        w1 = genome_data["controller"]["w1"]
        b1 = genome_data["controller"]["b1"]
        w2 = genome_data["controller"]["w2"]
        b2 = genome_data["controller"]["b2"]

        if TORCH_AVAILABLE and hasattr(controller, '_use_torch') and controller._use_torch:
            # Convert lists to tensors on the controller's device
            controller.w1 = torch.tensor(w1, dtype=torch.float32, device=controller.device)
            controller.b1 = torch.tensor(b1, dtype=torch.float32, device=controller.device)
            controller.w2 = torch.tensor(w2, dtype=torch.float32, device=controller.device)
            controller.b2 = torch.tensor(b2, dtype=torch.float32, device=controller.device)
        else:
            controller.w1 = w1
            controller.b1 = b1
            controller.w2 = w2
            controller.b2 = b2

        # Reconstruct genome
        genome = GENREGGenome(
            proteins=genome_data["proteins"],
            controller=controller
        )
        genome.id = genome_data["id"]
        genome.trust = genome_data["trust"]
        genome.food_eaten = genome_data.get("food_eaten", 0)
        genome.step_count = genome_data.get("step_count", 0)

        # Restore visual state if present
        visual_state = genome_data.get("visual_state", {})
        genome.correct_predictions = visual_state.get("correct_predictions", 0)
        genome.total_predictions = visual_state.get("total_predictions", 0)
        genome.consecutive_correct = visual_state.get("consecutive_correct", 0)

        # Restore stagnation state if present
        stagnation_state = genome_data.get("stagnation_state", {})
        genome.last_wrong_word = stagnation_state.get("last_wrong_word", "")
        genome.stagnation_count = stagnation_state.get("stagnation_count", 0)

        genomes.append(genome)
    
    # Reconstruct population
    population = GENREGPopulation(
        template_proteins=template_proteins,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        size=population_size,
        mutation_rate=0.1  # This will be overridden by evolve calls, so default is fine
    )
    
    # Replace genomes with loaded ones
    population.genomes = genomes
    population.active = checkpoint_data.get("active_index", 0)
    
    print(f"[CHECKPOINT] Loaded checkpoint from {checkpoint_path}")
    print(f"  Generation: {generation}")
    print(f"  Population size: {population_size}")
    print(f"  Best trust: {max(g.trust for g in genomes):.2f}")
    if phase_manager_state:
        print(f"  Phase: {phase_manager_state.get('current_phase', 1)}")

    return population, generation, template_proteins, phase_manager_state


def list_checkpoints(checkpoint_dir="checkpoints"):
    """
    List all available checkpoints.
    
    Returns:
        List of checkpoint file paths, sorted by generation
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_gen_") and filename.endswith(".pkl"):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            checkpoints.append(checkpoint_path)
    
    # Sort by generation number
    checkpoints.sort(key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]))
    
    return checkpoints


def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    """
    Get the path to the latest checkpoint.
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None



