"""
Advanced example demonstrating multi-dimensional MHW detection.

This example shows how to work with spatial datasets and demonstrates
the performance benefits of the xarray/dask-based implementation.
"""

import numpy as np
import xarray as xr
import pandas as pd
import time as time_module
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import marineheatwaves_xarray as mhw


def create_spatial_sst_data():
    """Create realistic spatial SST data with regional heat wave patterns."""
    
    # Create time coordinate (5 years of daily data)
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2004, 12, 31)
    time = pd.date_range(start_date, end_date, freq='D')
    n_time = len(time)
    
    # Create spatial coordinates (global grid)
    lat = np.arange(-70, 71, 2)  # 71 latitudes
    lon = np.arange(0, 360, 2)   # 180 longitudes
    
    print(f"Creating SST data with shape: ({n_time}, {len(lat)}, {len(lon)})")
    
    # Create base climatology with realistic spatial patterns
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    # Realistic SST spatial pattern
    # Higher temperatures at equator, lower at poles
    spatial_mean = 25 * np.cos(np.deg2rad(lat_grid)) + 5
    
    # Add longitudinal variation (warmer in western Pacific)
    lon_variation = 2 * np.sin(np.deg2rad(lon_grid * 2))
    spatial_mean += lon_variation
    
    # Create seasonal cycle
    day_of_year = np.array([t.timetuple().tm_yday for t in time])
    seasonal_amplitude = 5 * np.abs(np.cos(np.deg2rad(lat_grid)))  # Larger seasonal cycle away from equator
    seasonal_cycle = seasonal_amplitude[np.newaxis, :, :] * np.cos(
        2 * np.pi * (day_of_year[:, np.newaxis, np.newaxis] - 45) / 365.25
    )
    
    # Create base SST field
    sst_data = spatial_mean[np.newaxis, :, :] + seasonal_cycle
    
    # Add random variability
    np.random.seed(42)
    noise = np.random.normal(0, 0.8, sst_data.shape)
    sst_data += noise
    
    # Add regional marine heat wave events
    # Event 1: Pacific heat wave (El Niño-like pattern)
    pacific_lat_mask = (lat_grid >= -20) & (lat_grid <= 20)
    pacific_lon_mask = (lon_grid >= 120) & (lon_grid <= 280)
    pacific_mask = pacific_lat_mask & pacific_lon_mask
    
    # Strong heat wave in 2002 (days 730-780)
    hw_start = 730
    hw_end = 780
    for day in range(hw_start, hw_end):
        intensity = 4.0 * np.exp(-((day - (hw_start + hw_end) / 2) / 20) ** 2)  # Gaussian shape
        sst_data[day, pacific_mask] += intensity
    
    # Event 2: Atlantic heat wave
    atlantic_lat_mask = (lat_grid >= 20) & (lat_grid <= 60)
    atlantic_lon_mask = (lon_grid >= 280) | (lon_grid <= 40)
    atlantic_mask = atlantic_lat_mask & atlantic_lon_mask
    
    # Moderate heat wave in 2003 (days 1100-1130)
    hw_start2 = 1100
    hw_end2 = 1130
    for day in range(hw_start2, hw_end2):
        intensity = 3.0 * np.exp(-((day - (hw_start2 + hw_end2) / 2) / 15) ** 2)
        sst_data[day, atlantic_mask] += intensity
    
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
            'description': 'Synthetic global SST data with regional MHW events'
        }
    )
    
    return sst


def analyze_regional_mhws():
    """Analyze MHWs in different regions."""
    print("Creating spatial SST dataset...")
    sst = create_spatial_sst_data()
    
    print(f"Dataset shape: {sst.shape}")
    print(f"Memory usage: ~{sst.nbytes / 1e6:.1f} MB")
    
    # Define regions for analysis
    regions = {
        'Tropical_Pacific': {
            'lat_bounds': (-20, 20),
            'lon_bounds': (120, 280),
            'description': 'Tropical Pacific (ENSO region)'
        },
        'North_Atlantic': {
            'lat_bounds': (20, 60),
            'lon_bounds': (280, 40),
            'description': 'North Atlantic'
        },
        'Southern_Ocean': {
            'lat_bounds': (-60, -40),
            'lon_bounds': (0, 360),
            'description': 'Southern Ocean'
        }
    }
    
    results = {}
    
    for region_name, region_info in regions.items():
        print(f"\n--- Analyzing {region_name} ---")
        print(f"Description: {region_info['description']}")
        
        # Extract regional data
        lat_min, lat_max = region_info['lat_bounds']
        lon_min, lon_max = region_info['lon_bounds']
        
        if lon_max < lon_min:  # Handle longitude wraparound
            sst_region = sst.where(
                (sst.lat >= lat_min) & (sst.lat <= lat_max) &
                ((sst.lon >= lon_min) | (sst.lon <= lon_max)),
                drop=True
            )
        else:
            sst_region = sst.where(
                (sst.lat >= lat_min) & (sst.lat <= lat_max) &
                (sst.lon >= lon_min) & (sst.lon <= lon_max),
                drop=True
            )
        
        # Calculate regional average
        sst_regional_mean = sst_region.mean(dim=['lat', 'lon'], skipna=True)
        
        print(f"Regional SST range: {sst_regional_mean.min().values:.1f} to {sst_regional_mean.max().values:.1f} °C")
        
        # Detect MHWs in regional average
        start_time = time_module.time()
        
        try:
            mhw_events, climatology = mhw.detect(
                sst_regional_mean,
                climatology_period=(2000, 2002),  # Use first 3 years for climatology
                percentile=90.0,
                min_duration=5,
                window_half_width=5,
                smooth_percentile=True
            )
            
            detection_time = time_module.time() - start_time
            
            print(f"Detection completed in {detection_time:.3f} seconds")
            print(f"Detected {mhw_events['n_events']} marine heat wave events")
            
            if mhw_events['n_events'] > 0:
                print("\nEvent Summary:")
                for i in range(min(3, mhw_events['n_events'])):  # Show first 3 events
                    start_time_str = pd.to_datetime(mhw_events['time_start'][i]).strftime('%Y-%m-%d')
                    end_time_str = pd.to_datetime(mhw_events['time_end'][i]).strftime('%Y-%m-%d')
                    
                    print(f"  Event {i+1}: {start_time_str} to {end_time_str}")
                    print(f"    Duration: {mhw_events['duration'][i]} days")
                    print(f"    Max intensity: {mhw_events['intensity_max'][i]:.2f} °C")
                    print(f"    Category: {mhw_events['category'][i]}")
                
                # Calculate statistics
                block_stats = mhw.blockAverage(
                    sst_regional_mean.time,
                    mhw_events,
                    block_length=1
                )
                
                mean_vals, trends, trend_errors = mhw.meanTrend(block_stats)
                
                print(f"\nStatistics:")
                print(f"  Mean annual events: {mean_vals.get('count', 0):.2f}")
                print(f"  Mean event duration: {mean_vals.get('duration', 0):.1f} days")
                print(f"  Mean max intensity: {mean_vals.get('intensity_max', 0):.2f} °C")
                print(f"  Annual trend: {trends.get('count', 0):.3f} ± {trend_errors.get('count', 0):.3f} events/year")
                
                results[region_name] = {
                    'events': mhw_events,
                    'climatology': climatology,
                    'block_stats': block_stats,
                    'mean_values': mean_vals,
                    'trends': trends
                }
            else:
                print("No events detected in this region")
                results[region_name] = None
                
        except Exception as e:
            print(f"Error detecting MHWs in {region_name}: {e}")
            results[region_name] = None
    
    return results


def demonstrate_performance():
    """Demonstrate performance characteristics of the implementation."""
    print("\n" + "="*60)
    print("PERFORMANCE DEMONSTRATION")
    print("="*60)
    
    # Test different data sizes
    test_cases = [
        {'years': 5, 'lat_points': 36, 'lon_points': 72, 'description': '5°x5° global grid, 5 years'},
        {'years': 10, 'lat_points': 18, 'lon_points': 36, 'description': '10°x10° global grid, 10 years'},
        {'years': 3, 'lat_points': 72, 'lon_points': 144, 'description': '2.5°x2.5° global grid, 3 years'},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['description']}")
        
        # Create test data
        time = pd.date_range('2000-01-01', periods=case['years']*365, freq='D')
        lat = np.linspace(-90, 90, case['lat_points'])
        lon = np.linspace(0, 360, case['lon_points'])
        
        data_shape = (len(time), len(lat), len(lon))
        data_size_mb = np.prod(data_shape) * 8 / 1e6  # 8 bytes per float64
        
        print(f"  Data shape: {data_shape}")
        print(f"  Memory size: ~{data_size_mb:.1f} MB")
        
        # Create simple synthetic data
        np.random.seed(42)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        spatial_mean = 20 + 10 * np.cos(np.deg2rad(lat_grid))
        
        sst_data = spatial_mean[np.newaxis, :, :] + np.random.normal(0, 2, data_shape)
        
        sst = xr.DataArray(
            sst_data,
            coords={'time': time, 'lat': lat, 'lon': lon},
            dims=['time', 'lat', 'lon'],
            name='sst'
        )
        
        # Test single point extraction and detection
        start_time = time_module.time()
        
        # Extract a few representative points
        n_points = min(5, len(lat) * len(lon))
        sample_points = []
        
        for j in range(min(3, len(lat))):
            for k in range(min(2, len(lon))):
                sst_point = sst.isel(lat=j*len(lat)//3, lon=k*len(lon)//2)
                
                try:
                    mhw_events, _ = mhw.detect(
                        sst_point,
                        climatology_period=(2000, 2001),
                        percentile=90.0,
                        min_duration=5
                    )
                    sample_points.append(mhw_events['n_events'])
                except:
                    sample_points.append(0)
        
        detection_time = time.time() - start_time
        
        print(f"  Sample points processed: {len(sample_points)}")
        print(f"  Total events detected: {sum(sample_points)}")
        print(f"  Processing time: {detection_time:.3f} seconds")
        print(f"  Time per point: {detection_time/len(sample_points):.3f} seconds")


def main():
    """Main function for advanced example."""
    print("Advanced Marine Heat Wave Analysis")
    print("==================================")
    
    # Analyze regional MHWs
    regional_results = analyze_regional_mhws()
    
    # Demonstrate performance
    demonstrate_performance()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Summarize results
    total_events = 0
    for region_name, result in regional_results.items():
        if result:
            n_events = result['events']['n_events']
            total_events += n_events
            print(f"{region_name}: {n_events} events detected")
        else:
            print(f"{region_name}: No events or analysis failed")
    
    print(f"\nTotal events across all regions: {total_events}")
    print("\nAdvanced example completed successfully!")
    print("\nKey features demonstrated:")
    print("- Multi-dimensional data handling")
    print("- Regional analysis capabilities")
    print("- Performance with different data sizes")
    print("- Statistical analysis and trend calculation")


if __name__ == "__main__":
    main()