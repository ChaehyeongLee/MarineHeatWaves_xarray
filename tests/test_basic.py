"""
Basic tests for MarineHeatWaves_xarray package.

This module contains simple tests to validate the core functionality
of the marine heat wave detection algorithms.
"""

import numpy as np
import xarray as xr
import pandas as pd
import pytest
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import marineheatwaves_xarray as mhw


def create_test_sst_data():
    """Create simple test SST data with known heat wave events."""
    
    # Create 3 years of daily data
    time = pd.date_range('2000-01-01', '2002-12-31', freq='D')
    n_time = len(time)
    
    # Create simple temperature data with seasonal cycle
    day_of_year = np.array([t.timetuple().tm_yday for t in time])
    seasonal_cycle = 2.0 * np.cos(2 * np.pi * (day_of_year - 45) / 365.25)
    base_temp = 20.0 + seasonal_cycle
    
    # Add small random variations
    np.random.seed(42)  # For reproducible tests
    noise = np.random.normal(0, 0.5, n_time)
    
    # Add known heat wave events
    sst_data = base_temp + noise
    
    # Insert a strong heat wave event in summer 2001 (days 400-410)
    hw_start = 400
    hw_end = 410
    sst_data[hw_start:hw_end] += 5.0  # 5°C above normal
    
    # Insert a moderate heat wave event in summer 2002
    hw_start2 = 750
    hw_end2 = 755
    sst_data[hw_start2:hw_end2] += 3.0  # 3°C above normal
    
    # Create xarray DataArray
    sst = xr.DataArray(
        sst_data,
        coords={'time': time},
        dims=['time'],
        name='sst'
    )
    
    return sst


def test_package_import():
    """Test that the package imports correctly."""
    assert hasattr(mhw, 'detect')
    assert hasattr(mhw, 'blockAverage')
    assert hasattr(mhw, 'meanTrend')
    assert hasattr(mhw, 'rank')
    assert hasattr(mhw, '__version__')


def test_basic_detection():
    """Test basic MHW detection functionality."""
    sst = create_test_sst_data()
    
    # Detect MHWs using first 2 years for climatology
    mhw_events, climatology = mhw.detect(
        sst,
        climatology_period=(2000, 2001),
        percentile=90.0,
        min_duration=5
    )
    
    # Should detect at least the two events we inserted
    assert mhw_events['n_events'] >= 2
    
    # Check that we have the expected properties
    expected_keys = [
        'time_start', 'time_end', 'duration', 'intensity_max',
        'intensity_mean', 'category', 'n_events'
    ]
    for key in expected_keys:
        assert key in mhw_events
    
    # Check that climatology dataset has expected variables
    assert 'seas' in climatology
    assert 'thresh' in climatology
    assert 'missing' in climatology


def test_event_properties():
    """Test that detected events have reasonable properties."""
    sst = create_test_sst_data()
    
    mhw_events, _ = mhw.detect(
        sst,
        climatology_period=(2000, 2001),
        percentile=90.0,
        min_duration=5
    )
    
    if mhw_events['n_events'] > 0:
        # Check that durations are positive and >= min_duration
        for duration in mhw_events['duration']:
            assert duration >= 5
            
        # Check that intensities are positive (for heat waves)
        for intensity in mhw_events['intensity_max']:
            assert intensity > 0
            
        # Check that categories are valid
        valid_categories = ['Moderate', 'Strong', 'Severe', 'Extreme']
        for category in mhw_events['category']:
            assert category in valid_categories


def test_block_average():
    """Test block averaging functionality."""
    sst = create_test_sst_data()
    
    mhw_events, _ = mhw.detect(
        sst,
        climatology_period=(2000, 2001),
        percentile=90.0,
        min_duration=5
    )
    
    # Calculate annual block averages
    block_stats = mhw.blockAverage(
        sst.time,
        mhw_events,
        block_length=1
    )
    
    # Should have 3 blocks (3 years)
    assert len(block_stats['years_start']) == 3
    assert len(block_stats['count']) == 3
    
    # Check that counts are non-negative
    for count in block_stats['count']:
        assert count >= 0


def test_mean_trend():
    """Test mean and trend calculation."""
    sst = create_test_sst_data()
    
    mhw_events, _ = mhw.detect(
        sst,
        climatology_period=(2000, 2001),
        percentile=90.0,
        min_duration=5
    )
    
    block_stats = mhw.blockAverage(
        sst.time,
        mhw_events,
        block_length=1
    )
    
    mean_vals, trends, trend_errors = mhw.meanTrend(block_stats)
    
    # Should return dictionaries with expected keys
    assert isinstance(mean_vals, dict)
    assert isinstance(trends, dict)
    assert isinstance(trend_errors, dict)
    
    # Should have results for count
    assert 'count' in mean_vals
    assert 'count' in trends


def test_input_validation():
    """Test input validation functionality."""
    
    # Test with invalid input (not xarray DataArray)
    with pytest.raises(ValueError):
        mhw.detect([1, 2, 3, 4, 5])
    
    # Test with DataArray without time dimension
    invalid_data = xr.DataArray([1, 2, 3, 4, 5], dims=['x'])
    with pytest.raises(ValueError):
        mhw.detect(invalid_data)


def test_climatology_calculation():
    """Test climatology and threshold calculation."""
    sst = create_test_sst_data()
    
    _, climatology = mhw.detect(
        sst,
        climatology_period=(2000, 2001),
        percentile=90.0
    )
    
    # Climatology should have same time dimension as input
    assert climatology.seas.sizes['time'] == sst.sizes['time']
    assert climatology.thresh.sizes['time'] == sst.sizes['time']
    
    # Threshold should be higher than climatology on average
    assert (climatology.thresh.mean() > climatology.seas.mean()).item()


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    
    try:
        test_package_import()
        print("✓ Package import test passed")
        
        test_basic_detection()
        print("✓ Basic detection test passed")
        
        test_event_properties()
        print("✓ Event properties test passed")
        
        test_block_average()
        print("✓ Block average test passed")
        
        test_mean_trend()
        print("✓ Mean trend test passed")
        
        test_input_validation()
        print("✓ Input validation test passed")
        
        test_climatology_calculation()
        print("✓ Climatology calculation test passed")
        
        print("\nAll tests passed! ✓")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()