import os

def create_structure_in_current_dir():
    """Create the project structure in the current working directory"""
    
    # Define all directories to create
    directories = [
        "configs",
        "src/environment",
        "src/agents", 
        "src/policies",
        "src/data",
        "src/utils",
        "scripts",
        "notebooks",
        "data/raw",
        "data/processed",
        "results/figures",
        "results/tables", 
        "results/checkpoints",
        "results/logs",
        "tests"
    ]
    
    # Define all files to create
    files = [
        "README.md",
        "requirements.txt",
        "setup.py", 
        ".gitignore",
        
        "configs/__init__.py",
        "configs/hyperparameters.yaml",
        "configs/models_config.yaml",
        
        "src/__init__.py",
        
        "src/environment/__init__.py",
        "src/environment/base_economy.py",
        "src/environment/svar_economy.py", 
        "src/environment/ann_economy.py",
        
        "src/agents/__init__.py",
        "src/agents/ddpg_agent.py",
        "src/agents/networks.py",
        "src/agents/replay_buffer.py",
        
        "src/policies/__init__.py",
        "src/policies/linear_policy.py",
        "src/policies/nonlinear_policy.py",
        "src/policies/baseline_policies.py",
        
        "src/data/__init__.py", 
        "src/data/data_loader.py",
        "src/data/data_preparation.py",
        
        "src/utils/__init__.py",
        "src/utils/metrics.py",
        "src/utils/visualization.py",
        "src/utils/logger.py",
        
        "scripts/estimate_economy.py",
        "scripts/train_agent.py",
        "scripts/evaluate_policy.py",
        "scripts/static_counterfactual.py",
        "scripts/generate_figures.py",
        
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_economy_estimation.ipynb",
        "notebooks/03_rl_training.ipynb", 
        "notebooks/04_results_analysis.ipynb",
        
        "tests/__init__.py",
        "tests/test_environment.py",
        "tests/test_agent.py",
        "tests/test_policies.py"
    ]
    
    current_dir = os.getcwd()
    print(f"Creating project structure in current directory: {current_dir}")
    
    # Create all directories
    for directory in directories:
        dir_path = os.path.join(current_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create all files (empty)
    for file in files:
        file_path = os.path.join(current_dir, file)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create empty file
        with open(file_path, 'w') as f:
            pass  # Empty file
        
        print(f"Created file: {file}")
    
    print(f"\nProject structure created successfully in current directory!")

if __name__ == "__main__":
    create_structure_in_current_dir()