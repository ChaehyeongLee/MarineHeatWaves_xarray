"""
Utility functions for Marine Heat Wave detection using xarray and dask.

This module provides helper functions for data preprocessing, validation,
and climatology calculations that support the core MHW detection functionality.
"""

import numpy as np
import xarray as xr
import dask.array as da
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import warnings


def _validate_inputs(temp: xr.DataArray) -> xr.DataArray:
    """
    Validate input temperature data.
    
    Parameters
    ----------
    temp : xr.DataArray
        Temperature data to validate.
        
    Returns
    -------
    xr.DataArray
        Validated temperature data.
        
    Raises
    ------
    ValueError
        If validation fails.
    """
    if not isinstance(temp, xr.DataArray):
        raise ValueError("Temperature data must be an xarray.DataArray")
    
    if 'time' not in temp.dims:
        raise ValueError("Temperature data must have a 'time' dimension")
    
    if 'time' not in temp.coords:
        raise ValueError("Temperature data must have a 'time' coordinate")
    
    # Ensure time is datetime-like
    if not np.issubdtype(temp.time.dtype, np.datetime64):
        try:
            temp = temp.assign_coords(time=pd.to_datetime(temp.time))
        except Exception as e:
            raise ValueError(f"Could not convert time coordinate to datetime: {e}")
    
    # Sort by time if not already sorted
    if not temp.time.to_index().is_monotonic_increasing:
        temp = temp.sortby('time')
        warnings.warn("Time coordinate was not sorted. Data has been sorted by time.")
    
    return temp


def _extract_time_components(time_coord: xr.DataArray) -> Dict:
    """
    Extract time components (year, month, day, day-of-year) from time coordinate.
    
    Parameters
    ----------
    time_coord : xr.DataArray
        Time coordinate array.
        
    Returns
    -------
    dict
        Dictionary containing time components.
    """
    time_info = {}
    
    # Convert to pandas datetime index for easier manipulation
    time_index = pd.to_datetime(time_coord.values)
    
    time_info['year'] = xr.DataArray(time_index.year, coords={'time': time_coord}, dims=['time'])
    time_info['month'] = xr.DataArray(time_index.month, coords={'time': time_coord}, dims=['time'])
    time_info['day'] = xr.DataArray(time_index.day, coords={'time': time_coord}, dims=['time'])
    time_info['dayofyear'] = xr.DataArray(time_index.dayofyear, coords={'time': time_coord}, dims=['time'])
    
    return time_info


def calculate_climatology(
    temp: xr.DataArray,
    climatology_period: Tuple[int, int],
    window_half_width: int = 5,
    smooth: bool = True,
    smooth_width: int = 31,
    non_standard_calendar: bool = False,
) -> xr.DataArray:
    """
    Calculate seasonal climatology using a moving window approach.
    
    Parameters
    ----------
    temp : xr.DataArray
        Temperature data for climatology calculation.
    climatology_period : tuple of int
        Start and end years for climatology period.
    window_half_width : int, default 5
        Half-width of window around each day-of-year.
    smooth : bool, default True
        Whether to apply smoothing to the climatology.
    smooth_width : int, default 31
        Width of smoothing window.
    non_standard_calendar : bool, default False
        Whether calendar has non-standard length.
        
    Returns
    -------
    xr.DataArray
        Seasonal climatology for each day of year.
    """
    # Extract time components
    time_info = _extract_time_components(temp.time)
    
    # Filter to climatology period
    clim_mask = (
        (time_info['year'] >= climatology_period[0]) & 
        (time_info['year'] <= climatology_period[1])
    )
    temp_clim_period = temp.where(clim_mask, drop=True)
    time_info_clim = {
        key: val.where(clim_mask, drop=True) 
        for key, val in time_info.items()
    }
    
    # Determine calendar length
    if non_standard_calendar:
        # For simplified implementation, assume 360-day calendar
        calendar_length = 360
        doy_values = np.arange(1, calendar_length + 1)
    else:
        calendar_length = 366  # Use leap year as reference
        doy_values = np.arange(1, calendar_length + 1)
    
    # Initialize climatology array
    clim_shape = [calendar_length] + list(temp.shape[1:])  # [day_of_year, ...spatial_dims]
    clim_data = np.full(clim_shape, np.nan)
    
    # Calculate climatology for each day of year
    for doy in doy_values:
        # Handle Feb 29 specially
        if doy == 60 and not non_standard_calendar:  # Feb 29
            continue
            
        # Find all days within window around this day-of-year
        window_doys = []
        for offset in range(-window_half_width, window_half_width + 1):
            target_doy = doy + offset
            
            # Handle year wraparound
            if target_doy < 1:
                target_doy += calendar_length
            elif target_doy > calendar_length:
                target_doy -= calendar_length
                
            window_doys.append(target_doy)
        
        # Find matching days in the data
        doy_mask = xr.concat([
            time_info_clim['dayofyear'] == target_doy 
            for target_doy in window_doys
        ], dim='time').any(dim='time')
        
        if doy_mask.any():
            temp_window = temp_clim_period.where(doy_mask)
            clim_data[doy - 1] = temp_window.mean(dim='time', skipna=True)
    
    # Handle Feb 29 by interpolation
    if not non_standard_calendar:
        feb28_idx, feb29_idx, mar1_idx = 58, 59, 60  # 0-based indices
        if not np.isnan(clim_data[feb28_idx]).all() and not np.isnan(clim_data[mar1_idx]).all():
            clim_data[feb29_idx] = 0.5 * (clim_data[feb28_idx] + clim_data[mar1_idx])
    
    # Create output DataArray with proper coordinates
    doy_coord = xr.DataArray(doy_values, dims=['dayofyear'], name='dayofyear')
    
    # Create coordinates dictionary
    coords = {'dayofyear': doy_coord}
    spatial_coords = {k: v for k, v in temp.coords.items() if k != 'time'}
    coords.update(spatial_coords)
    
    # Create output DataArray
    clim_array = xr.DataArray(
        clim_data,
        dims=['dayofyear'] + list(temp.dims[1:]),
        coords=coords,
        name='climatology'
    )
    
    # Apply smoothing if requested
    if smooth:
        clim_array = _smooth_climatology(clim_array, smooth_width, non_standard_calendar)
    
    # Map back to original time coordinate
    time_doy = _extract_time_components(temp.time)['dayofyear']
    
    # Handle leap years by mapping Feb 29 to interpolated value
    doy_mapped = time_doy.copy()
    if not non_standard_calendar:
        # Map Feb 29 (doy 60) to our interpolated value
        doy_mapped = xr.where(doy_mapped > 366, 366, doy_mapped)
    
    clim_timeseries = clim_array.sel(dayofyear=doy_mapped, method='nearest')
    clim_timeseries = clim_timeseries.assign_coords(time=temp.time)
    
    return clim_timeseries


def calculate_threshold(
    temp: xr.DataArray,
    climatology_period: Tuple[int, int],
    percentile: float = 90.0,
    window_half_width: int = 5,
    smooth: bool = True,
    smooth_width: int = 31,
    non_standard_calendar: bool = False,
) -> xr.DataArray:
    """
    Calculate threshold percentiles using a moving window approach.
    
    Parameters
    ----------
    temp : xr.DataArray
        Temperature data for threshold calculation.
    percentile : float, default 90.0
        Percentile to calculate for threshold.
    climatology_period : tuple of int
        Start and end years for climatology period.
    window_half_width : int, default 5
        Half-width of window around each day-of-year.
    smooth : bool, default True
        Whether to apply smoothing to the threshold.
    smooth_width : int, default 31
        Width of smoothing window.
    non_standard_calendar : bool, default False
        Whether calendar has non-standard length.
        
    Returns
    -------
    xr.DataArray
        Threshold percentiles for each day of year.
    """
    # Extract time components
    time_info = _extract_time_components(temp.time)
    
    # Filter to climatology period
    clim_mask = (
        (time_info['year'] >= climatology_period[0]) & 
        (time_info['year'] <= climatology_period[1])
    )
    temp_clim_period = temp.where(clim_mask, drop=True)
    time_info_clim = {
        key: val.where(clim_mask, drop=True) 
        for key, val in time_info.items()
    }
    
    # Determine calendar length
    if non_standard_calendar:
        calendar_length = 360
        doy_values = np.arange(1, calendar_length + 1)
    else:
        calendar_length = 366
        doy_values = np.arange(1, calendar_length + 1)
    
    # Initialize threshold array
    thresh_shape = [calendar_length] + list(temp.shape[1:])
    thresh_data = np.full(thresh_shape, np.nan)
    
    # Calculate threshold for each day of year
    for doy in doy_values:
        # Handle Feb 29 specially
        if doy == 60 and not non_standard_calendar:
            continue
            
        # Find all days within window around this day-of-year
        window_doys = []
        for offset in range(-window_half_width, window_half_width + 1):
            target_doy = doy + offset
            
            # Handle year wraparound
            if target_doy < 1:
                target_doy += calendar_length
            elif target_doy > calendar_length:
                target_doy -= calendar_length
                
            window_doys.append(target_doy)
        
        # Find matching days in the data
        doy_mask = xr.concat([
            time_info_clim['dayofyear'] == target_doy 
            for target_doy in window_doys
        ], dim='time').any(dim='time')
        
        if doy_mask.any():
            temp_window = temp_clim_period.where(doy_mask)
            thresh_data[doy - 1] = temp_window.quantile(
                percentile / 100.0, dim='time', skipna=True
            )
    
    # Handle Feb 29 by interpolation
    if not non_standard_calendar:
        feb28_idx, feb29_idx, mar1_idx = 58, 59, 60
        if not np.isnan(thresh_data[feb28_idx]).all() and not np.isnan(thresh_data[mar1_idx]).all():
            thresh_data[feb29_idx] = 0.5 * (thresh_data[feb28_idx] + thresh_data[mar1_idx])
    
    # Create output DataArray
    doy_coord = xr.DataArray(doy_values, dims=['dayofyear'], name='dayofyear')
    coords = {'dayofyear': doy_coord}
    spatial_coords = {k: v for k, v in temp.coords.items() if k != 'time'}
    coords.update(spatial_coords)
    
    thresh_array = xr.DataArray(
        thresh_data,
        dims=['dayofyear'] + list(temp.dims[1:]),
        coords=coords,
        name='threshold'
    )
    
    # Apply smoothing if requested
    if smooth:
        thresh_array = _smooth_climatology(thresh_array, smooth_width, non_standard_calendar)
    
    # Map back to original time coordinate
    time_doy = _extract_time_components(temp.time)['dayofyear']
    doy_mapped = time_doy.copy()
    if not non_standard_calendar:
        doy_mapped = xr.where(doy_mapped > 366, 366, doy_mapped)
    
    thresh_timeseries = thresh_array.sel(dayofyear=doy_mapped, method='nearest')
    thresh_timeseries = thresh_timeseries.assign_coords(time=temp.time)
    
    return thresh_timeseries


def _smooth_climatology(
    data: xr.DataArray, 
    smooth_width: int, 
    non_standard_calendar: bool = False
) -> xr.DataArray:
    """
    Apply smoothing to climatological data using a running mean.
    
    Parameters
    ----------
    data : xr.DataArray
        Data to smooth along dayofyear dimension.
    smooth_width : int
        Width of smoothing window.
    non_standard_calendar : bool
        Whether calendar has non-standard length.
        
    Returns
    -------
    xr.DataArray
        Smoothed data.
    """
    # Ensure smooth_width is odd
    if smooth_width % 2 == 0:
        smooth_width += 1
    
    half_width = smooth_width // 2
    
    # For periodic smoothing, we need to pad the data
    if non_standard_calendar:
        # For 360-day calendar, simple periodic extension
        data_extended = xr.concat([
            data.isel(dayofyear=slice(-half_width, None)),
            data,
            data.isel(dayofyear=slice(0, half_width))
        ], dim='dayofyear')
    else:
        # For standard calendar, handle the fact that we have 366 days
        # Skip NaN values (which might exist for days not in all years)
        valid_mask = ~data.isnull()
        if valid_mask.all():
            # Simple case: extend periodically
            data_extended = xr.concat([
                data.isel(dayofyear=slice(-half_width, None)),
                data,
                data.isel(dayofyear=slice(0, half_width))
            ], dim='dayofyear')
        else:
            # More complex case: handle missing values
            data_extended = data  # Simplified for now
    
    # Apply rolling mean
    smoothed = data_extended.rolling(
        dayofyear=smooth_width, 
        center=True, 
        min_periods=1
    ).mean()
    
    # Extract the central portion
    if non_standard_calendar or valid_mask.all():
        smoothed = smoothed.isel(dayofyear=slice(half_width, -half_width))
    else:
        smoothed = data  # Fallback to original data
    
    return smoothed


def pad_missing_values(temp: xr.DataArray, max_pad_length: int) -> xr.DataArray:
    """
    Interpolate missing values in temperature data.
    
    Parameters
    ----------
    temp : xr.DataArray
        Temperature data with potential missing values.
    max_pad_length : int
        Maximum consecutive length of missing values to interpolate.
        
    Returns
    -------
    xr.DataArray
        Temperature data with interpolated values.
    """
    # Use xarray's interpolate_na method
    temp_filled = temp.interpolate_na(
        dim='time',
        method='linear',
        max_gap=pd.Timedelta(days=max_pad_length),
        keep_attrs=True
    )
    
    return temp_filled


def runavg(data: np.ndarray, window_width: int) -> np.ndarray:
    """
    Calculate running average with periodic boundary conditions.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array.
    window_width : int
        Width of running average window (should be odd).
        
    Returns
    -------
    np.ndarray
        Smoothed data array.
    """
    if window_width % 2 == 0:
        window_width += 1
    
    # Create periodic extension
    half_width = window_width // 2
    data_extended = np.concatenate([
        data[-half_width:],
        data,
        data[:half_width]
    ])
    
    # Apply convolution
    kernel = np.ones(window_width) / window_width
    smoothed_extended = np.convolve(data_extended, kernel, mode='same')
    
    # Extract central portion
    smoothed = smoothed_extended[half_width:-half_width]
    
    return smoothed


def create_chunks_dict(data: xr.DataArray, chunk_size: str = "auto") -> Dict:
    """
    Create appropriate chunking dictionary for dask operations.
    
    Parameters
    ----------
    data : xr.DataArray
        Input data array.
    chunk_size : str or dict, default "auto"
        Chunking specification.
        
    Returns
    -------
    dict
        Chunking dictionary optimized for MHW calculations.
    """
    if chunk_size == "auto":
        # Optimize chunking for time series operations
        chunks = {}
        
        # Keep time dimension unchunked for efficient time series operations
        chunks['time'] = -1
        
        # Chunk spatial dimensions reasonably
        for dim in data.dims:
            if dim != 'time':
                size = data.sizes[dim]
                if size > 100:
                    chunks[dim] = min(50, size // 4)
                else:
                    chunks[dim] = -1
    else:
        chunks = chunk_size
    
    return chunks


def validate_climatology_period(
    temp: xr.DataArray, 
    climatology_period: Optional[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Validate and adjust climatology period based on available data.
    
    Parameters
    ----------
    temp : xr.DataArray
        Temperature data.
    climatology_period : tuple of int or None
        Requested climatology period.
        
    Returns
    -------
    tuple of int
        Validated climatology period.
        
    Raises
    ------
    ValueError
        If climatology period is invalid.
    """
    # Extract year range from data
    time_info = _extract_time_components(temp.time)
    data_start_year = int(time_info['year'].min())
    data_end_year = int(time_info['year'].max())
    
    if climatology_period is None:
        return data_start_year, data_end_year
    
    clim_start, clim_end = climatology_period
    
    # Validate climatology period
    if clim_start > clim_end:
        raise ValueError("Climatology start year must be <= end year")
    
    if clim_start < data_start_year or clim_end > data_end_year:
        raise ValueError(
            f"Climatology period ({clim_start}-{clim_end}) extends beyond "
            f"available data ({data_start_year}-{data_end_year})"
        )
    
    if clim_end - clim_start + 1 < 10:
        warnings.warn(
            f"Climatology period is quite short ({clim_end - clim_start + 1} years). "
            "Consider using a longer period for more robust statistics."
        )
    
    return clim_start, clim_end