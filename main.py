#!/usr/bin/env python3
"""
Main entry point for Monetary Policy RL project.

This script provides a unified interface to run all project tasks:
- Data preparation and preprocessing
- Economy estimation (SVAR and ANN)
- Agent training (all configurations)
- Policy evaluation (counterfactual analysis)
- Figure and table generation
- Complete pipeline execution

Usage:
    python main.py --help
    python main.py --task all
    python main.py --task estimate --model ann
    python main.py --task train --economy ann --policy nonlinear --lags 1
    python main.py --task evaluate --mode historical
    python main.py --task figures --figure 2
 
Author: Monetary Policy RL Team
Date: 2024
"""

import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import logging
from pathlib import Path
from typing import Optional, List
import subprocess
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger


class ProjectRunner:
    """
    Main project runner class for executing all tasks.
    
    This class provides a unified interface for running different components
    of the Monetary Policy RL project with flexible configuration.
    """
    
    def __init__(
        self,
        config_path: str = "configs/hyperparameters.yaml",
        verbose: bool = True
    ):
        """
        Initialize project runner.
        
        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
        """
        self.config_path = config_path
        self.verbose = verbose
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        log_level = logging.INFO if verbose else logging.WARNING
        self.logger = setup_logger(
            'main',
            'results/logs/main.log',
            level=log_level
        )
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary project directories."""
        directories = [
            'data/raw',
            'data/processed',
            'results/figures',
            'results/tables',
            'results/checkpoints',
            'results/logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _run_script(
        self,
        script_path: str,
        args: List[str],
        description: str
    ) -> int:
        """
        Run a Python script with arguments.
        
        Args:
            script_path: Path to script
            args: List of command-line arguments
            description: Task description for logging
        
        Returns:
            Exit code
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"{description}")
        self.logger.info(f"{'='*60}")
        
        cmd = [sys.executable, script_path] + args
        
        if self.verbose:
            self.logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=not self.verbose,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"[OK] {description} completed successfully")
            
            return result.returncode
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[FAILED] {description} failed with error code {e.returncode}")
            if e.stderr:
                self.logger.error(f"Error: {e.stderr}")
            return e.returncode
        except Exception as e:
            self.logger.error(f"[FAILED] {description} failed: {str(e)}")
            return 1
    
    def prepare_data(self) -> int:
        """
        Download and prepare US macroeconomic data.
        
        Returns:
            Exit code
        """
        return self._run_script(
            'scripts/data_preparation.py',
            [],
            "DATA PREPARATION"
        )
    
    def estimate_economy(
        self,
        model: str = 'both',
        output_dir: Optional[str] = None,
        tune: bool = False,
        trials: int = 20,
        epochs: int = 500
    ) -> int:
        """
        Estimate economy models (SVAR and/or ANN).
        
        Args:
            model: Model to estimate ('svar', 'ann', 'both')
            output_dir: Custom output directory
            tune: Tune hyperparameters before estimation
            trials: Number of tuning trials
            epochs: Max epochs for tuning
        
        Returns:
            Exit code
        """
        args = [
            '--model', model,
            '--config', self.config_path
        ]
        
        if tune:
            args.append('--tune')
            args.extend(['--trials', str(trials)])
            args.extend(['--epochs', str(epochs)])
        
        if output_dir:
            args.extend(['--output_dir', output_dir])
        
        return self._run_script(
            'scripts/estimate_economy.py',
            args,
            f"ECONOMY ESTIMATION: {model.upper()}"
        )
    
    def train_agent(
        self,
        economy: str,
        policy: str,
        lags: int,
        device: str = 'cpu',
        checkpoint_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> int:
        """
        Train a single DDPG agent.
        
        Args:
            economy: Economy type ('svar' or 'ann')
            policy: Policy type ('linear' or 'nonlinear')
            lags: Include lags (0 or 1)
            device: Computation device ('cpu' or 'cuda')
            checkpoint_dir: Directory with economy checkpoints
            output_dir: Output directory for trained agent
        
        Returns:
            Exit code
        """
        args = [
            '--economy', economy,
            '--policy', policy,
            '--lags', str(lags),
            '--device', device,
            '--config', self.config_path
        ]
        
        if checkpoint_dir:
            args.extend(['--checkpoint_dir', checkpoint_dir])
        if output_dir:
            args.extend(['--output_dir', output_dir])
        
        return self._run_script(
            'scripts/train_agent.py',
            args,
            f"TRAINING: {economy.upper()} + {policy.upper()} + LAGS={lags}"
        )
    
    def train_all_agents(
        self,
        device: str = 'cpu',
        skip_svar_nonlinear: bool = True
    ) -> int:
        """
        Train all agent configurations.
        
        Args:
            device: Computation device
            skip_svar_nonlinear: Skip SVAR nonlinear (not in paper)
        
        Returns:
            Exit code (0 if all succeed)
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING ALL AGENTS")
        self.logger.info("="*60)
        
        configurations = [
            ('svar', 'linear', 0),
            ('svar', 'linear', 1),
            ('ann', 'linear', 0),
            ('ann', 'linear', 1),
            ('ann', 'nonlinear', 0),
            ('ann', 'nonlinear', 1),
        ]
        
        if not skip_svar_nonlinear:
            configurations.extend([
                ('svar', 'nonlinear', 0),
                ('svar', 'nonlinear', 1),
            ])
        
        total = len(configurations)
        failed = []
        
        for idx, (economy, policy, lags) in enumerate(configurations, 1):
            self.logger.info(f"\n[{idx}/{total}] Training {economy.upper()} + "
                           f"{policy.upper()} + LAGS={lags}")
            
            result = self.train_agent(economy, policy, lags, device)
            
            if result != 0:
                failed.append((economy, policy, lags))
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total configurations: {total}")
        self.logger.info(f"Successful: {total - len(failed)}")
        self.logger.info(f"Failed: {len(failed)}")
        
        if failed:
            self.logger.error("\nFailed configurations:")
            for economy, policy, lags in failed:
                self.logger.error(f"  - {economy} + {policy} + LAGS={lags}")
            return 1
        else:
            self.logger.info("\n[OK] All agents trained successfully!")
            return 0
    
    def evaluate_policies(
        self,
        mode: str = 'both',
        economy: str = 'ann',
        checkpoint_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> int:
        """
        Evaluate policies using counterfactual analysis.
        
        Args:
            mode: Evaluation mode ('historical', 'static', 'both')
            economy: Economy to use for evaluation
            checkpoint_dir: Directory with trained agents
            output_dir: Output directory
        
        Returns:
            Exit code
        """
        args = [
            '--mode', mode,
            '--economy', economy,
            '--config', self.config_path
        ]
        
        if checkpoint_dir:
            args.extend(['--checkpoint_dir', checkpoint_dir])
        if output_dir:
            args.extend(['--output_dir', output_dir])
        
        return self._run_script(
            'scripts/evaluate_policy.py',
            args,
            f"POLICY EVALUATION: {mode.upper()}"
        )
    
    def generate_figures(
        self,
        figure_numbers: Optional[List[int]] = None,
        all_figures: bool = False,
        tables: bool = False,
        learning_curve: bool = False,
        checkpoint_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> int:
        """
        Generate figures and tables from paper.
        
        Args:
            figure_numbers: Specific figure numbers to generate
            all_figures: Generate all figures
            tables: Generate tables
            learning_curve: Generate learning curves
            checkpoint_dir: Directory with checkpoints
            output_dir: Output directory
        
        Returns:
            Exit code
        """
        args = ['--config', self.config_path]
        
        if all_figures:
            args.append('--all')
        elif figure_numbers:
            for fig_num in figure_numbers:
                args.extend(['--figure', str(fig_num)])
        
        if tables:
            args.append('--tables')
            
        if learning_curve:
            args.append('--learning_curve')
        
        if checkpoint_dir:
            args.extend(['--checkpoint_dir', checkpoint_dir])
        if output_dir:
            args.extend(['--output_dir', output_dir])
        
        return self._run_script(
            'scripts/generate_figures.py',
            args,
            "FIGURE GENERATION"
        )
    
    def run_complete_pipeline(
        self,
        device: str = 'cpu',
        skip_data: bool = False
    ) -> int:
        """
        Run complete pipeline from data to figures.
        
        Args:
            device: Computation device
            skip_data: Skip data preparation (if already done)
        
        Returns:
            Exit code
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("RUNNING COMPLETE PIPELINE")
        self.logger.info("="*80)
        
        steps = []
        
        # Step 1: Data preparation
        if not skip_data:
            self.logger.info("\n[Step 1/5] Data Preparation")
            result = self.prepare_data()
            steps.append(('Data Preparation', result))
            if result != 0:
                self.logger.error("Pipeline failed at data preparation")
                return result
        
        # Step 2: Economy estimation
        self.logger.info("\n[Step 2/5] Economy Estimation")
        result = self.estimate_economy('both')
        steps.append(('Economy Estimation', result))
        if result != 0:
            self.logger.error("Pipeline failed at economy estimation")
            return result
        
        # Step 3: Agent training
        self.logger.info("\n[Step 3/5] Agent Training")
        result = self.train_all_agents(device)
        steps.append(('Agent Training', result))
        if result != 0:
            self.logger.error("Pipeline failed at agent training")
            return result
        
        # Step 4: Policy evaluation
        self.logger.info("\n[Step 4/5] Policy Evaluation")
        result = self.evaluate_policies('both', 'ann')
        steps.append(('Policy Evaluation', result))
        if result != 0:
            self.logger.error("Pipeline failed at policy evaluation")
            return result
        
        # Step 5: Figure generation
        self.logger.info("\n[Step 5/5] Figure Generation")
        result = self.generate_figures(all_figures=True, tables=True)
        steps.append(('Figure Generation', result))
        if result != 0:
            self.logger.error("Pipeline failed at figure generation")
            return result
        
        # Final summary
        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("="*80)
        
        for step_name, step_result in steps:
            status = "[OK]" if step_result == 0 else "[FAILED]"
            self.logger.info(f"{status} {step_name}")
        
        if all(result == 0 for _, result in steps):
            self.logger.info("\n[SUCCESS] Complete pipeline finished successfully!")
            return 0
        else:
            self.logger.error("\n[FAILED] Pipeline completed with errors")
            return 1
    
    def clean_results(
        self,
        figures: bool = False,
        checkpoints: bool = False,
        logs: bool = False,
        all_results: bool = False
    ):
        """
        Clean generated results.
        """
        import shutil
        import logging
        from pathlib import Path
        
        # Close all loggers and use print for cleanup
        logging.shutdown()
        
        print("\n" + "="*60)
        print("CLEANING RESULTS")
        print("="*60)
        
        if all_results or figures:
            print("Cleaning figures...")
            figures_dir = Path('results/figures')
            if figures_dir.exists():
                for pattern in ['*.pdf', '*.png', '*.jpg']:
                    for file in figures_dir.glob(pattern):
                        try:
                            file.unlink()
                            print(f"  Removed {file}")
                        except Exception as e:
                            print(f"  Could not remove {file}: {e}")
        
        if all_results or checkpoints:
            print("Cleaning checkpoints...")
            checkpoints_dir = Path('results/checkpoints')
            if checkpoints_dir.exists():
                for pattern in ['*.pth', '*.pkl', '*.pt']:
                    for file in checkpoints_dir.glob(pattern):
                        try:
                            file.unlink()
                            print(f"  Removed {file}")
                        except Exception as e:
                            print(f"  Could not remove {file}: {e}")
        
        if all_results or logs:
            print("Cleaning logs...")
            logs_dir = Path('results/logs')
            if logs_dir.exists():
                for pattern in ['*.log', '*.csv', '*.txt']:
                    for file in logs_dir.glob(pattern):
                        try:
                            file.unlink()
                            print(f"  Removed {file}")
                        except Exception as e:
                            print(f"  Could not remove {file}: {e}")
        
        print("[OK] Cleaning completed")
        
        # Reinitialize the original logger by calling the existing setup method
        # This assumes you have some logger setup in your __init__ method
        # If not, the next operation will set it up again
    
    def status(self):
        """
        Display project status and available artifacts.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("PROJECT STATUS")
        self.logger.info("="*60)
        
        # Check data
        data_path = Path('data/processed/us_macro_data.csv')
        if data_path.exists():
            self.logger.info("[OK] Data: Available")
        else:
            self.logger.info("[FAILED] Data: Not found (run: python main.py --task data)")
        
        # Check economy models
        svar_path = Path('results/checkpoints/svar_params.pkl')
        ann_y_path = Path('results/checkpoints/ann_y_network.pth')
        ann_pi_path = Path('results/checkpoints/ann_pi_network.pth')
        
        if svar_path.exists():
            self.logger.info("[OK] SVAR Economy: Estimated")
        else:
            self.logger.info("[FAILED] SVAR Economy: Not estimated")
        
        if ann_y_path.exists() and ann_pi_path.exists():
            self.logger.info("[OK] ANN Economy: Estimated")
        else:
            self.logger.info("[FAILED] ANN Economy: Not estimated")
        
        # Check trained agents
        agent_patterns = [
            'ddpg_svar_linear_lags0.pth',
            'ddpg_svar_linear_lags1.pth',
            'ddpg_ann_linear_lags0.pth',
            'ddpg_ann_linear_lags1.pth',
            'ddpg_ann_nonlinear_lags0.pth',
            'ddpg_ann_nonlinear_lags1.pth',
        ]
        
        trained_agents = 0
        for pattern in agent_patterns:
            if (Path('results/checkpoints') / pattern).exists():
                trained_agents += 1
        
        self.logger.info(f"[OK] Trained Agents: {trained_agents}/{len(agent_patterns)}")
        
        # Check figures
        figure_count = len(list(Path('results/figures').glob('*.pdf')))
        self.logger.info(f"[OK] Generated Figures: {figure_count}")
        
        # Check tables
        table_count = len(list(Path('results/tables').glob('*.csv')))
        self.logger.info(f"[OK] Generated Tables: {table_count}")
        
        self.logger.info("\n" + "="*60)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Monetary Policy RL - Main Project Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --task all
  
  # Prepare data only
  python main.py --task data
  
  # Estimate both economies
  python main.py --task estimate --model both
  
  # Train specific agent
  python main.py --task train --economy ann --policy nonlinear --lags 1
  
  # Train all agents
  python main.py --task train-all --device cuda
  
  # Evaluate policies
  python main.py --task evaluate --mode historical
  
  # Generate specific figure
  python main.py --task figures --figure 2
  
  # Generate all figures and tables
  python main.py --task figures --all --tables
  
  # Check project status
  python main.py --task status
  
  # Clean results
  python main.py --task clean --all
        """
    )
    
    # Main arguments
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=[
            'all', 'data', 'estimate', 'train', 'train-all',
            'evaluate', 'figures', 'status', 'clean'
        ],
        help='Task to execute'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/hyperparameters.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Computation device'
    )
    
    # Estimation arguments
    parser.add_argument(
        '--model',
        type=str,
        choices=['svar', 'ann', 'both'],
        default='both',
        help='Model to estimate (for estimate task)'
    )
    
    # Training arguments
    parser.add_argument(
        '--economy',
        type=str,
        choices=['svar', 'ann'],
        help='Economy type (for train task)'
    )
    
    parser.add_argument(
        '--policy',
        type=str,
        choices=['linear', 'nonlinear'],
        help='Policy type (for train task)'
    )
    
    parser.add_argument(
        '--lags',
        type=int,
        choices=[0, 1],
        help='Include lags: 0=no, 1=yes (for train task)'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['historical', 'static', 'both'],
        default='both',
        help='Evaluation mode (for evaluate task)'
    )
    
    # Figure generation arguments
    parser.add_argument(
        '--figure',
        type=int,
        action='append',
        help='Figure number to generate (can specify multiple)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all figures (for figures task)'
    )
    
    parser.add_argument(
        '--tables',
        action='store_true',
        help='Generate tables (for figures task)'
    )
    
    parser.add_argument(
        '--learning_curve',
        action='store_true',
        help='Generate learning curves (for figures task)'
    )
    
    # Clean arguments
    parser.add_argument(
        '--figures-only',
        action='store_true',
        help='Clean only figures (for clean task)'
    )
    
    parser.add_argument(
        '--checkpoints-only',
        action='store_true',
        help='Clean only checkpoints (for clean task)'
    )
    
    parser.add_argument(
        '--logs-only',
        action='store_true',
        help='Clean only logs (for clean task)'
    )
    
    # Directory arguments
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Custom checkpoint directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory'
    )
    
    # Pipeline arguments
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data preparation in pipeline (for all task)'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create runner
    runner = ProjectRunner(
        config_path=args.config,
        verbose=args.verbose
    )
    
    # Execute task
    try:
        if args.task == 'all':
            # Run complete pipeline
            exit_code = runner.run_complete_pipeline(
                device=args.device,
                skip_data=args.skip_data
            )
        
        elif args.task == 'data':
            # Prepare data
            exit_code = runner.prepare_data()
        
        elif args.task == 'estimate':
            # Estimate economy
            exit_code = runner.estimate_economy(
                model=args.model,
                output_dir=args.output_dir
            )
        
        elif args.task == 'train':
            # Train single agent
            if not all([args.economy, args.policy, args.lags is not None]):
                parser.error("--task train requires --economy, --policy, and --lags")
            
            exit_code = runner.train_agent(
                economy=args.economy,
                policy=args.policy,
                lags=args.lags,
                device=args.device,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir
            )
        
        elif args.task == 'train-all':
            # Train all agents
            exit_code = runner.train_all_agents(device=args.device)
        
        elif args.task == 'evaluate':
            # Evaluate policies
            exit_code = runner.evaluate_policies(
                mode=args.mode,
                economy=args.economy or 'ann',
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir
            )
        
        elif args.task == 'figures':
            # Generate figures
            exit_code = runner.generate_figures(
                figure_numbers=args.figure,
                all_figures=args.all,
                tables=args.tables,
                learning_curve=args.learning_curve,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir
            )
        
        elif args.task == 'status':
            # Show status
            runner.status()
            exit_code = 0
        
        elif args.task == 'clean':
            # Clean results
            runner.clean_results(
                figures=args.figures_only or args.all,
                checkpoints=args.checkpoints_only or args.all,
                logs=args.logs_only or args.all,
                all_results=args.all
            )
            exit_code = 0
        
        else:
            parser.error(f"Unknown task: {args.task}")
            exit_code = 1
        
        sys.exit(exit_code)
    
    except KeyboardInterrupt:
        runner.logger.warning("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        runner.logger.error(f"\n\nFatal error: {str(e)}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()