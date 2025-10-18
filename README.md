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
git clone https://github.com/yourusername/monetary_policy_rl.git
cd monetary_policy_rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### 1. Data Preparation

Download and preprocess US macroeconomic data (1987:Q3 - 2023:Q2):
```bash
python scripts/data_preparation.py
```

This downloads:
- GDP Deflator (inflation)
- Output Gap (CBO estimates)
- Federal Funds Rate + Wu-Xia Shadow Rate

### 2. Estimate Economy Models

Estimate both SVAR and ANN transition equations:
```bash
# Estimate linear SVAR economy
python scripts/estimate_economy.py --model svar

# Estimate nonlinear ANN economy
python scripts/estimate_economy.py --model ann
```

**Outputs:**
- `results/checkpoints/svar_params.pkl`
- `results/checkpoints/ann_y_network.pth`
- `results/checkpoints/ann_pi_network.pth`
- `results/tables/table2_economy_mse.csv`

### 3. Train RL Agents

Train DDPG agents for all six configurations:
```bash
# Train all agents (takes ~2-4 hours on CPU)
python scripts/train_agent.py --economy ann --policy linear --lags 0
python scripts/train_agent.py --economy ann --policy linear --lags 1
python scripts/train_agent.py --economy ann --policy nonlinear --lags 0
python scripts/train_agent.py --economy ann --policy nonlinear --lags 1
python scripts/train_agent.py --economy svar --policy linear --lags 0
python scripts/train_agent.py --economy svar --policy linear --lags 1

# Or train all at once
bash scripts/train_all.sh
```

**Outputs:**
- `results/checkpoints/ddpg_agent_*.pth`
- `results/logs/training_*.csv`
- `results/tables/table3_policy_parameters.tex`

### 4. Run Counterfactual Analysis

Evaluate trained policies against baselines:
```bash
# Historical counterfactual (dynamic simulation)
python scripts/evaluate_policy.py --mode historical

# Static counterfactual
python scripts/evaluate_policy.py --mode static
```

**Outputs:**
- `results/tables/table4_counterfactual_loss.csv`
- Counterfactual time series data

### 5. Generate Figures

Reproduce all figures from the paper:
```bash
python scripts/generate_figures.py --all

# Or generate individual figures
python scripts/generate_figures.py --figure 2  # Economy fit comparison
python scripts/generate_figures.py --figure 6  # Historical counterfactual
```

**Outputs:** All figures in `results/figures/` (PDF, 300 DPI)

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