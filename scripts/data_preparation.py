#!/usr/bin/env python3
"""
Data preparation script.

Downloads and preprocesses US macroeconomic data for the project.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger


def main():
    """Main data preparation function."""
    logger = setup_logger('data_preparation', 'results/logs/data_preparation.log')
    
    logger.info("="*60)
    logger.info("DATA PREPARATION")
    logger.info("="*60)
    
    try:
        # Load and prepare data
        data_loader = DataLoader(
            start_date="1987-07-01",
            end_date="2023-06-30",
            data_dir="data/raw"
        )
        
        # Export for estimation
        data_loader.export_for_estimation(output_dir="data/processed")
        
        logger.info("\n✓ Data preparation completed successfully!")
        logger.info(f"  Data saved to: data/processed/")
        
        return 0
    
    except Exception as e:
        logger.error(f"\n✗ Data preparation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())