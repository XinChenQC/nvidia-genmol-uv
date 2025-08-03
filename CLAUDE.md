# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for generative fragment-based molecule design using NVIDIA's GenMol model. The project implements discrete diffusion for drug discovery tasks including linker design, motif extension, scaffold decoration, and superstructure generation.

## Development Environment Setup

This project uses **uv** (from Astral) as the package manager instead of pip or conda.

### Installation Commands
```bash
# Install dependencies and create virtual environment
uv sync

# Install in development mode (if needed)
uv pip install -e .
```

### Required Model File
The project requires a pre-trained model checkpoint that must be downloaded:
```bash
curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/genmol_v1/1.0/files?redirect=true&path=model.ckpt' -o model.ckpt
```

## Running the Code

### Command Line Interface
The main entry point is `main.py` which provides a CLI for molecule generation tasks:

```bash
# Example usage with uv
uv run python main.py --task linker_design --fragment "[11*]N1CCCC1.[17*]c1ccc2c(c1)OCCO2" --num-samples 20

# Supported tasks:
# - linker_design: Generate linkers between two fragments
# - motif_extension: Extend a molecular motif
# - scaffold_decoration: Add decorations to scaffolds
# - superstructure_generation: Generate full molecules from partial structures
# - random: De novo molecule generation
```

### Python API Usage
```python
from genmol.sampler import Sampler

sampler = Sampler('./model.ckpt')
samples = sampler.fragment_linking(fragment, num_samples=20, randomness=3)
```

## Code Architecture

### Core Components

1. **`genmol/model.py`**: Contains `GenMol` class - PyTorch Lightning module implementing the discrete diffusion model using BERT backbone and MDLM (Masked Diffusion Language Model)

2. **`genmol/sampler.py`**: Contains `Sampler` class - Main interface for molecule generation with methods for different tasks:
   - `de_novo_generation()`: Random molecule generation
   - `fragment_linking()`: Two-step linker design 
   - `fragment_linking_onestep()`: Single-step linker design
   - `fragment_completion()`: Motif extension/scaffold decoration/superstructure generation

3. **`genmol/utils/`**: Utility modules for:
   - `utils_chem.py`: Chemical processing (SMILES/SAFE conversions)
   - `utils_data.py`: Data handling and tokenization
   - `bracket_safe_converter.py`: SAFE string format conversions
   - `ema.py`: Exponential moving average for training

4. **`main.py`**: CLI wrapper with task-specific parameter defaults

### Key Technical Details

- Uses SAFE (Sequential Attachment-based Fragment Embedding) string representation for molecules
- Implements discrete diffusion via masking/unmasking tokens
- Model checkpoint contains both regular weights and EMA (Exponential Moving Average) weights
- Supports classifier-free guidance via gamma parameter for fragment conditioning
- Uses PyTorch Lightning for training framework
- Tokenization handled via HuggingFace transformers

### Fragment Input Format
Fragments use SMILES with attachment points marked as `[*]`, `[1*]`, `[2*]`, etc. Examples:
- `[1*]c1ccccc1` - benzene with one attachment point
- `[1*]N1CCCC1.[2*]c1ccc2c(c1)OCCO2` - two separate fragments for linking

### Default Parameters by Task
- **linker_design**: `gamma=0, randomness=3, softmax_temp=1.2`
- **motif_extension**: `gamma=0.3, randomness=1.2, softmax_temp=1.2`  
- **scaffold_decoration**: `gamma=0.3, randomness=2, softmax_temp=1.2`
- **superstructure_generation**: `gamma=0.4, randomness=2, softmax_temp=1.2`
- **random**: `gamma=0, randomness=0.5, softmax_temp=0.5`

## Dependencies

Main dependencies (see `pyproject.toml`):
- `torch>=2.7.1` - PyTorch framework
- `lightning>=2.5.2` - PyTorch Lightning 
- `transformers>=4.50.3` - HuggingFace transformers (BERT)
- `bionemo-moco>=0.0.2.1` - NVIDIA's molecular diffusion library
- `safe-mol>=0.1.13` - SAFE string molecular representation
- `openbabel-wheel>=3.1.1.22` - Chemical informatics

## File Structure Notes

- No traditional test files found in repository
- Uses `pyproject.toml` for dependency management (not `requirements.txt`)
- Model weights not included in repository (must be downloaded separately)
- Uses `uv.lock` for dependency version locking