"""
Tests for data loading and preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.data_loader import (
    DataLoader,
    check_stationarity,
    prepare_training_data,
    create_lagged_features
)


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        dates = pd.date_range('1987-07-01', periods=100, freq='Q')
        data = pd.DataFrame({
            'inflation': 2.0 + np.random.randn(100) * 0.5,
            'output_gap': 0.0 + np.random.randn(100) * 1.0,
            'interest_rate': 3.0 + np.random.randn(100) * 0.5
        }, index=dates)
        return data
    
    def test_check_stationarity(self, mock_data):
        """Test stationarity checks."""
        results = check_stationarity(
            mock_data['inflation'],
            name='Inflation',
            verbose=False
        )
        
        assert 'adf_stationary' in results
        assert 'kpss_stationary' in results
        assert 'both_agree' in results
        assert isinstance(results['adf_stationary'], bool)
    
    def test_prepare_training_data(self, mock_data):
        """Test train/validation split."""
        train_df, val_df = prepare_training_data(
            mock_data,
            validation_split=0.15,
            seed=42
        )
        
        # Check sizes
        assert len(train_df) + len(val_df) == len(mock_data)
        assert len(val_df) == int(len(mock_data) * 0.15)
        
        # Check no overlap
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        assert len(train_indices.intersection(val_indices)) == 0
    
    def test_create_lagged_features(self, mock_data):
        """Test creating lagged features."""
        lagged_df = create_lagged_features(mock_data, lags=2)
        
        # Check columns exist
        assert 'inflation_lag1' in lagged_df.columns
        assert 'inflation_lag2' in lagged_df.columns
        assert 'output_gap_lag1' in lagged_df.columns
        
        # Check values
        assert lagged_df['inflation_lag1'].iloc[2] == mock_data['inflation'].iloc[1]
        assert lagged_df['inflation_lag2'].iloc[2] == mock_data['inflation'].iloc[0]
        
        # Check no NaN
        assert not lagged_df.isnull().any().any()
        
        # Should be shorter due to dropping NaN
        assert len(lagged_df) == len(mock_data) - 2


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""
    
    def test_lagged_features_different_lags(self):
        """Test lagged features with different lag lengths."""
        dates = pd.date_range('2000-01-01', periods=20, freq='Q')
        data = pd.DataFrame({
            'x': np.arange(20)
        }, index=dates)
        
        lagged_1 = create_lagged_features(data, lags=1)
        lagged_3 = create_lagged_features(data, lags=3)
        
        assert len(lagged_1) == 19
        assert len(lagged_3) == 17
        
        assert 'x_lag1' in lagged_1.columns
        assert 'x_lag3' in lagged_3.columns
        assert 'x_lag3' not in lagged_1.columns
    
    def test_lagged_features_values(self):
        """Test that lagged values are correct."""
        dates = pd.date_range('2000-01-01', periods=10, freq='Q')
        data = pd.DataFrame({
            'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }, index=dates)
        
        lagged = create_lagged_features(data, lags=2)
        
        # Check specific values
        assert lagged['x'].iloc[0] == 2
        assert lagged['x_lag1'].iloc[0] == 1
        assert lagged['x_lag2'].iloc[0] == 0


class TestDataQuality:
    """Tests for data quality checks."""
    
    def test_no_missing_values(self, sample_data):
        """Test that processed data has no missing values."""
        lagged = create_lagged_features(sample_data, lags=2)
        
        assert not lagged.isnull().any().any()
    
    def test_date_range(self, sample_data):
        """Test that date range is correct."""
        assert sample_data.index.freq == 'Q'
        assert len(sample_data) == 100
    
    def test_variable_ranges(self, sample_data):
        """Test that variables are in reasonable ranges."""
        # Inflation should be positive
        assert (sample_data['inflation'] > -5).all()
        assert (sample_data['inflation'] < 20).all()
        
        # Output gap should be bounded
        assert (sample_data['output_gap'] > -10).all()
        assert (sample_data['output_gap'] < 10).all()
        
        # Interest rate should be positive (mostly)
        assert (sample_data['interest_rate'] > -5).all()