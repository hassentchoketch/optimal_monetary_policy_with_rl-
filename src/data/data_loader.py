"""
Data loading and preprocessing for US macroeconomic data.

Downloads quarterly data from FRED:
- GDP Deflator (inflation)
- Output Gap (CBO estimates)
- Federal Funds Rate
- Wu-Xia Shadow Rate
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import os
from statsmodels.tsa.stattools import adfuller, kpss


def load_us_data(
    start_date: str = "1987-07-01",
    end_date: str = "2023-06-30",
    data_dir: str = "data\\raw",
    save_processed: bool = True
) -> pd.DataFrame:
    """
    Load and preprocess US macroeconomic data.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_dir: Directory containing raw data
        save_processed: Whether to save processed data
    
    Returns:
        DataFrame with columns: ['inflation', 'output_gap', 'interest_rate']
    """
    # Note: In practice, use pandas_datareader or fredapi
    # For this implementation, we assume CSV files are provided
    
    try:
        # Try to load from FRED using pandas_datareader
        import pandas_datareader as pdr
        
        start_date: str = "1987-07-01"
        end_date: str = "2023-06-30"

        # GDP Deflator (for inflation calculation)
        gdp_deflator = pdr.get_data_fred('GDPDEF', start=start_date, end=end_date)

        # Calculate year-over-year inflation
        inflation = gdp_deflator.pct_change(periods=4) * 100
        inflation.columns = ['inflation']

        # Output gap (using CBO potential GDP)
        gdp = pdr.get_data_fred('GDPC1', start=start_date, end=end_date)
        potential_gdp = pdr.get_data_fred('GDPPOT', start=start_date, end=end_date)
        output_gap = ((gdp["GDPC1"] - potential_gdp["GDPPOT"]) / potential_gdp["GDPPOT"] * 100).reset_index().set_index('DATE')
        output_gap.columns = ['output_gap']

        # Federal Funds Rate
        ffr = pdr.get_data_fred('DFF', start=start_date, end=end_date)

        # Quarterly average of monthly data
        ffr_quarterly = ffr.resample('QS').mean()
        ffr_quarterly.columns = ['interest_rate']
        
        # Load Wu-Xia shadow rate for ZLB period
        # This would need to be downloaded separately from:
        # https://www.atlantafed.org/cqer/research/wu-xia-shadow-federal-funds-rate
        # shadow_rate_path = os.path.join(data_dir, 'wu_xia_shadow_rate.csv')
        
        # if os.path.exists(shadow_rate_path):
        #     shadow_rate = pd.read_csv(shadow_rate_path, index_col=0, parse_dates=True)
        #     shadow_rate_quarterly = shadow_rate.resample('QE').mean()
            
        #     # Replace FFR with shadow rate when FFR is at ZLB (< 0.25%)
        #     mask = ffr_quarterly['interest_rate'] < 0.25
        #     ffr_quarterly.loc[mask, 'interest_rate'] = shadow_rate_quarterly.loc[mask].values
        
        # Combine all series
        df = pd.concat([inflation, output_gap, ffr_quarterly], axis=1)
        df = df.dropna()
        
        # Ensure quarterly frequency
        df = df.asfreq('QS')
        
    except Exception as e:
        print(f"Error loading from FRED: {e}")
        print("Loading from provided CSV files...")
        
        # Fallback: load from CSV files
        inflation_path = os.path.join(data_dir, 'inflation.csv')
        output_gap_path = os.path.join(data_dir, 'output_gap.csv')
        ffr_path = os.path.join(data_dir, 'federal_funds_rate.csv')
        
        inflation = pd.read_csv(inflation_path, index_col=0, parse_dates=True)
        output_gap = pd.read_csv(output_gap_path, index_col=0, parse_dates=True)
        ffr = pd.read_csv(ffr_path, index_col=0, parse_dates=True)
        
        df = pd.concat([inflation, output_gap, ffr], axis=1)
        df.columns = ['inflation', 'output_gap', 'interest_rate']
        df = df.loc[start_date:end_date]
    
    # Save processed data
    if save_processed:
        processed_dir = "data\\processed"
        os.makedirs(processed_dir, exist_ok=True)
        df.to_csv(os.path.join(processed_dir, 'us_macro_data.csv'))
    
    return df

def check_stationarity(
    data: pd.Series,
    name: str = "Series",
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Perform stationarity tests (ADF and KPSS).
    
    As mentioned in the paper (footnote 15), all series should be stationary.
    
    Args:
        data: Time series data
        name: Name of series
        verbose: Print results
    
    Returns:
        Dictionary with test results
    """
    # Augmented Dickey-Fuller test
    # H0: Series has unit root (non-stationary)
    adf_result = adfuller(data.dropna(), autolag='AIC')
    adf_stationary = adf_result[1] < 0.05  # p-value < 0.05
    
    # KPSS test
    # H0: Series is stationary
    kpss_result = kpss(data.dropna(), regression='c', nlags='auto')
    kpss_stationary = kpss_result[1] > 0.05  # p-value > 0.05
    
    if verbose:
        print(f"\nStationarity tests for {name}:")
        print(f"  ADF test: statistic={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")
        print(f"  -> {'Stationary' if adf_stationary else 'Non-stationary'}")
        print(f"  KPSS test: statistic={kpss_result[0]:.4f}, p-value={kpss_result[1]:.4f}")
        print(f"  -> {'Stationary' if kpss_stationary else 'Non-stationary'}")
    
    return {
        'adf_stationary': adf_stationary,
        'kpss_stationary': kpss_stationary,
        'both_agree': adf_stationary == kpss_stationary
    }

def prepare_training_data(
    df: pd.DataFrame,
    validation_split: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time series data chronologically into training and validation sets.

    This is the standard and correct method for time series data, as it ensures
    the model is validated on data that occurs "after" the training data,
    preventing data leakage from the future.

    Args:
        df: Full dataset, which should be sorted chronologically.
        validation_split: Fraction of the data to use for the validation set,
                          which will be taken from the end of the dataset.

    Returns:
        A tuple containing the training DataFrame and the validation DataFrame.
        (train_df, val_df)
    """
    # Ensure the DataFrame is sorted by index (time) before splitting
    df = df.sort_index()

    n_total = len(df)
    n_train = int(n_total * (1 - validation_split))

    # The training set is the first part of the data
    train_df = df.iloc[:n_train]

    # The validation set is the final part of the data
    val_df = df.iloc[n_train:]

    return train_df, val_df

def create_lagged_features(
    df: pd.DataFrame,
    lags: int = 2
) -> pd.DataFrame:
    """
    Create lagged features for SVAR estimation.
    
    Args:
        df: DataFrame with ['inflation', 'output_gap', 'interest_rate']
        lags: Number of lags
    
    Returns:
        DataFrame with lagged features
    """
    df_lagged = df.copy()
    
    for var in df.columns:
        for lag in range(1, lags + 1):
            df_lagged[f'{var}_lag{lag}'] = df[var].shift(lag)
    
    # Drop rows with NaN (from lagging)
    df_lagged = df_lagged.dropna()
    
    return df_lagged

class DataLoader:
    """
    Data loader class for easy access to US macroeconomic data.
    """
    
    def __init__(
        self,
        start_date: str = "1987-07-01",
        end_date: str = "2023-06-30",
        data_dir: str = "data/raw"
    ):
        """
        Initialize data loader.
        
        Args:
            start_date: Start date
            end_date: End date
            data_dir: Directory for raw data
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        
        # Load data
        self.df = load_us_data(start_date, end_date, data_dir)
        print(f"Data loaded: {len(self.df)} observations from {self.df.index.min().date()} to {self.df.index.max().date()}")
        # Perform stationarity checks
        print("\n" + "="*60)
        print("STATIONARITY CHECKS")
        print("="*60)
        for col in self.df.columns:
            check_stationarity(self.df[col], name=col)
    
    def get_data(self) -> pd.DataFrame:
        """Get full dataset."""
        return self.df.copy()
    
    def get_train_val_split(
        self,
        validation_split: float = 0.15,
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train/validation split."""
        return prepare_training_data(self.df, validation_split, seed)
    
    def get_lagged_data(self, lags: int = 2) -> pd.DataFrame:
        """Get data with lagged features."""
        return create_lagged_features(self.df, lags)
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics."""
        return self.df.describe()
    
    def export_for_estimation(
        self,
        output_dir: str = "data/processed"
    ):
        """
        Export data in format needed for economy estimation.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full data
        self.df.to_csv(os.path.join(output_dir, 'us_macro_data.csv'))
        
        # Save train/val split for ANN
        train_df, val_df = self.get_train_val_split()
        train_df.to_csv(os.path.join(output_dir, 'train_data.csv'))
        val_df.to_csv(os.path.join(output_dir, 'val_data.csv'))
        
        # Save lagged data for SVAR
        lagged_df = self.get_lagged_data(lags=2)
        lagged_df.to_csv(os.path.join(output_dir, 'lagged_data.csv'))
        
        print(f"\nData exported to {output_dir}")
        print(f"  - Full data: {len(self.df)} observations")
        print(f"  - Training: {len(train_df)} observations")
        print(f"  - Validation: {len(val_df)} observations")