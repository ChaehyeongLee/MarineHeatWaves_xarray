"""
Example usage of MarineHeatWaves_xarray package.

This example demonstrates how to use the package with synthetic data
to detect marine heat waves and calculate their properties.
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import marineheatwaves_xarray as mhw


def create_synthetic_sst_data():
    """Create synthetic SST data for demonstration."""
    
    # Create time coordinate (10 years of daily data)
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2009, 12, 31)
    time = pd.date_range(start_date, end_date, freq='D')
    
    # Create spatial coordinates
    lat = np.arange(-40, 41, 5)  # 17 latitudes
    lon = np.arange(0, 360, 5)   # 72 longitudes
    
    # Create base climatology with seasonal cycle
    n_time = len(time)
    day_of_year = np.array([t.timetuple().tm_yday for t in time])
    
    # Seasonal cycle (cosine)
    seasonal_cycle = 2.0 * np.cos(2 * np.pi * (day_of_year - 45) / 365.25)
    
    # Add spatial variation
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    spatial_mean = 15 + 10 * np.cos(np.deg2rad(lat_grid))
    
    # Create full SST field
    sst_data = np.zeros((n_time, len(lat), len(lon)))
    
    for i in range(n_time):
        # Base temperature with seasonal cycle
        base_temp = spatial_mean + seasonal_cycle[i]
        
        # Add some random variability
        noise = np.random.normal(0, 1, spatial_mean.shape)
        
        # Add occasional heat waves (random spikes)
        if np.random.random() < 0.1:  # 10% chance of heat wave
            heat_wave = np.random.uniform(3, 8, spatial_mean.shape)
            # Apply heat wave to random subset of points
            hw_mask = np.random.random(spatial_mean.shape) < 0.3
            noise[hw_mask] += heat_wave[hw_mask]
        
        sst_data[i] = base_temp + noise
    
    # Create xarray DataArray
    sst = xr.DataArray(
        sst_data,
        coords={
            'time': time,
            'lat': lat,
            'lon': lon
        },
        dims=['time', 'lat', 'lon'],
        name='sst',
        attrs={
            'units': 'degrees_C',
            'long_name': 'Sea Surface Temperature',
            'description': 'Synthetic SST data for MHW demonstration'
        }
    )
    
    return sst


def main():
    """Main example function."""
    
    print("Creating synthetic SST data...")
    sst = create_synthetic_sst_data()
    
    print(f"SST data shape: {sst.shape}")
    print(f"Time range: {sst.time.values[0]} to {sst.time.values[-1]}")
    print(f"Spatial range: lat {sst.lat.values.min():.1f} to {sst.lat.values.max():.1f}, "
          f"lon {sst.lon.values.min():.1f} to {sst.lon.values.max():.1f}")
    
    # For demonstration, select a single grid point
    print("\nSelecting single grid point for detailed analysis...")
    sst_point = sst.isel(lat=8, lon=36)  # Mid-latitude point
    
    print("Detecting marine heat waves...")
    try:
        mhw_events, climatology = mhw.detect(
            sst_point,
            climatology_period=(2000, 2005),  # Use first 6 years for climatology
            percentile=90.0,
            min_duration=5,
            window_half_width=5,
            smooth_percentile=True
        )
        
        print(f"Detected {mhw_events['n_events']} marine heat wave events")
        
        if mhw_events['n_events'] > 0:
            print("\nMHW Event Summary:")
            for i in range(min(5, mhw_events['n_events'])):  # Show first 5 events
                print(f"Event {i+1}:")
                print(f"  Duration: {mhw_events['duration'][i]} days")
                print(f"  Max intensity: {mhw_events['intensity_max'][i]:.2f} °C")
                print(f"  Mean intensity: {mhw_events['intensity_mean'][i]:.2f} °C")
                print(f"  Category: {mhw_events['category'][i]}")
        
        # Calculate block averages
        print("\nCalculating annual statistics...")
        block_stats = mhw.blockAverage(
            sst_point.time, 
            mhw_events, 
            block_length=1  # Annual blocks
        )
        
        print(f"Annual event counts: {block_stats['count']}")
        
        # Calculate trends
        print("\nCalculating trends...")
        mean_vals, trends, trend_errors = mhw.meanTrend(block_stats)
        
        print(f"Mean annual count: {mean_vals.get('count', np.nan):.2f}")
        print(f"Count trend: {trends.get('count', np.nan):.3f} events/year")
        
    except Exception as e:
        print(f"Error during MHW detection: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()