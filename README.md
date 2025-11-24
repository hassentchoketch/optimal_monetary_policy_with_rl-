# Optimal Monetary Policy using Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Replication code for "Optimal Monetary Policy using Reinforcement Learning" by Hinterlang & Tänzer (2024).**

This project leverages **Deep Deterministic Policy Gradient (DDPG)** reinforcement learning to discover optimal interest rate reaction functions for central banks. By training agents in both linear (SVAR) and nonlinear (ANN) economic environments, The Paper demonstrate that AI-driven policies can significantly outperform historical benchmarks.

---

## ✨ Key Features

- 🤖 **DDPG Agent**: Advanced actor-critic architecture for continuous action spaces.
- 🌍 **Dual Economies**: Train in both Linear SVAR and Nonlinear ANN environments.
- 📈 **Counterfactual Analysis**: rigorous historical and static evaluations.
- ⚡ **Optimized Performance**: GPU-accelerated training with PyTorch.
- 📊 **Comprehensive Visualization**: Automated generation of publication-ready figures.

## 🚀 Key Results

The RL agents consistently outperform historical policy decisions:

> [!IMPORTANT]
> **27% Loss Reduction**: The nonlinear RL policy achieves a **27% reduction in central bank loss** compared to actual Federal Reserve decisions (1987-2023).

See the results for yourself:
- [Counterfactual Analysis (RL)](results/figures/figure7_counterfactual_rl.pdf)
- [Economy Fit](results/figures/figure2_economy_fit.pdf)

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional)

### Quick Setup
```bash
# Clone and install
git clone https://github.com/hassentchoketch/optimal_monetary_policy_with_rl.git
cd optimal_monetary_policy_with_rl

# Create venv and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

## 🏁 Quick Start

The `main.py` script is your command center.

### Option A: Full Replication (Paper Results)
Run the entire pipeline from data preparation to figure generation:
```bash
python main.py --task all
```
*Note: This may take 2-4 hours depending on your hardware.*

### Option B: Train a Single Agent
Experiment with a specific configuration:
```bash
python main.py --task train --economy ann --policy nonlinear --lags 1
```

### Option C: Generate Figures
Regenerate all visualizations from existing results:
```bash
python main.py --task figures --all --tables
```

## 📂 Project Structure

```
monetary_policy_rl/
├── src/
│   ├── agents/             # DDPG implementation (Actor/Critic)
│   ├── environment/        # Economy models (SVAR, ANN)
│   ├── policies/           # Policy rules (Taylor, RL, etc.)
│   └── data/               # Data loading & preprocessing
├── scripts/                # Execution scripts
├── configs/                # Hyperparameters (YAML)
├── results/                # Generated figures, tables, & checkpoints
└── tests/                  # Unit and integration tests
```

## 🛠️ Development

### Running Tests
Ensure everything is working correctly with `pytest`:

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only
pytest tests/ -v -m "not slow"
```

### Configuration
Adjust hyperparameters in `configs/hyperparameters.yaml`.
```yaml
rl_training:
  episodes: 500
  learning_rate: 2.5e-5
  batch_size: 64
```
## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---
*Original paper by Hinterlang & Tänzer (2024). Implementation by Hacene Tchoketch kebir 