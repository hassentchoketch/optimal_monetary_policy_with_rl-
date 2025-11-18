# Optimal Monetary Policy using Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Replication code for **"Optimal Monetary Policy using Reinforcement Learning"** by Hinterlang & Tänzer (2024).

## Overview

This project implements a Deep Deterministic Policy Gradient (DDPG) reinforcement learning approach to compute optimal interest rate reaction functions for central banks. The implementation includes:

- **Two economy representations**: Linear SVAR and Nonlinear ANN
- **Six RL-optimized policy rules**: Linear and nonlinear variants
- **Complete counterfactual analysis**: Historical and static evaluations
- **Baseline comparisons**: Taylor (1993), NPP, Balanced Approach rules

### Key Results

- RL-optimized linear policies reduce central bank loss by **15%**
- Nonlinear policies achieve **27%+ loss reduction**
- Policies outperform actual Federal Reserve decisions (1987-2023)

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/optimal_monetary_policy_with_rl.git
cd optimal_monetary_policy_with_rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```
## Quick Start with main.py

The `main.py` script provides a unified interface for all project tasks:

### Complete Pipeline
```bash
# Run everything (load and preprocssed data  → estimate → train → evaluate → figures)
python main.py --task all

# Skip data download and preprocessed if already available
python main.py --task all --skip-data

# Use GPU for training
python main.py --task all --device cuda
```

### Individual Tasks

#### 1. Data  Preparation
```bash
python main.py --task data
```

#### 2. Economy Estimation
```bash
# Estimate both SVAR and ANN
python main.py --task estimate --model both

# Estimate only ANN
python main.py --task estimate --model ann
```

#### 3. Agent Training
```bash
# Train specific configuration
python main.py --task train --economy ann --policy nonlinear --lags 1

# Train all configurations
python main.py --task train-all

# Train with GPU
python main.py --task train-all --device cuda
```

#### 4. Policy Evaluation
```bash
# Both historical and static counterfactuals
python main.py --task evaluate --mode both

# Only historical counterfactual
python main.py --task evaluate --mode historical --economy ann
```

#### 5. Figure Generation
```bash
# Generate all figures and tables
python main.py --task figures --all --tables

# Generate specific figure
python main.py --task figures --figure 2
python main.py --task figures --figure 6 --figure 7
```

### Utility Commands

#### Check Project Status
```bash
python main.py --task status
```

#### Clean Results
```bash
# Clean everything
python main.py --task clean --all

# Clean only figures
python main.py --task clean --figures-only

# Clean only checkpoints
python main.py --task clean --checkpoints-only
```

### Advanced Usage

#### Custom Directories
```bash
python main.py --task train \
  --economy ann --policy linear --lags 0 \
  --checkpoint-dir custom/checkpoints \
  --output-dir custom/output
```

#### Verbose Mode
```bash
python main.py --task all --verbose
```

#### Help
```bash
python main.py --help
```

### Common Workflows

**Quick Test Run:**
```bash
# Estimate economies only
python main.py --task estimate --model both

# Train one agent for testing
python main.py --task train --economy ann --policy linear --lags 0
```

**Paper Replication:**
```bash
# Full replication (takes 2-4 hours)
python main.py --task all
```

**Update Figures After Changes:**
```bash
# Regenerate all visualizations
python main.py --task figures --all --tables
```

## Project Structure
```
monetary_policy_rl/
├── src/
│   ├── environment/        # Economy models (SVAR, ANN)
│   ├── agents/             # DDPG implementation
│   ├── policies/           # Policy rules (baseline + RL)
│   ├── data/               # Data loading and preprocessing
│   └── utils/              # Metrics, visualization, logging
├── scripts/                # Execution scripts
├── configs/                # Hyperparameters
├── data/                   # Raw and processed data
├── results/                # Outputs (figures, tables, checkpoints)
└── tests/                  # Unit tests
```

## Configuration

All hyperparameters are defined in `configs/hyperparameters.yaml`. Key parameters:
```yaml
rl_training:
  episodes: 500
  max_steps_per_episode: 12
  discount_factor: 0.99
  learning_rate: 2.5e-5
  batch_size: 64
  buffer_size: 10000
  
reward_function:
  weight_inflation: 0.5
  weight_output_gap: 0.5
  target_inflation: 2.0
  penalty_threshold: 2.0
```

## Reproducibility

All experiments use fixed random seeds for reproducibility:
```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```

To replicate exact results:
1. Use the same data vintage (files provided in `data/raw/`)
2. Run with default hyperparameters
3. Use PyTorch 2.0+ and NumPy 1.24+

## Results Summary

### Table 2: Economy Fit (MSE)

| Model | Output Gap | Inflation | Total |
|-------|-----------|-----------|-------|
| SVAR  | 0.935     | 0.090     | 0.512 |
| ANN   | **0.748** | **0.065** | **0.407** |

### Table 4: Counterfactual Loss (vs. Actual)

| Policy | Loss | Improvement |
|--------|------|-------------|
| Actual | 3.03 | - |
| TR93   | 2.95 | 3% |
| RL_ANN,no_lag | 2.71 | **11%** |
| RL_ANN,one_lag | 2.59 | **15%** |
| RL_ANN,no_lag,nonlin | 2.29 | **24%** |
| RL_ANN,one_lag,nonlin | **2.21** | **27%** |

## Citation

If you use this code, please cite:
```bibtex
@article{hinterlang2024optimal,
  title={Optimal Monetary Policy using Reinforcement Learning},
  author={Hinterlang, Natascha and T{\"a}nzer, Alina},
  journal={Economic Modelling},
  year={2024},
  note={Preprint available at SSRN 4977979}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your.email@example.com]

## Acknowledgments

- Original paper: Hinterlang & Tänzer (2024)
- DDPG algorithm: Lillicrap et al. (2015)
- Data sources: FRED, CBO, Wu-Xia Shadow Rate