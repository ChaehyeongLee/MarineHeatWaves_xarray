import numpy as np
import xarray as xr
import datetime
import pandas as pd
from scipy import ndimage
'''
This is for getting Marine heatwave properties from sea temperature. It can be done with n-dimensional datasets (e.g. (time,lat,lat,depth) corrdinate)

For the best performance
- The chunk size of the datasets should be 1/10 to 1/15 of the available memory of the worker.
- Define an optimal chunk size after setting the largest available 'time' coordinate at first.

'''

def _fill_missing(ds):
    '''
    Filling missing values within 'time' coordinate with np.NaN and removing Feb 29th
    '''
    t0 = ds.time.isel(time=0).values
    t1 = ds.time.isel(time=-1).values
    
    full_time = pd.date_range(t0,t1,freq='D')
    full_time = full_time[~((full_time.month ==2) & (full_time.day==29))]
    
    ds = ds.reindex(time=full_time)
    return ds
    
def _add_doy_coord(ds):
    '''
    Add new day-of-year coordinate to avoid mismatch caused by removal of leap day
    '''
    ds = _fill_missing(ds)
    key = ds.time.dt.strftime('%m-%d')
    ds  = ds.assign_coords(
            doy=('time', pd.to_datetime('2001-' + key.values).dayofyear.astype('int16'))
          )
    return ds

def _adjacent_doy_sel(ds, window=25):
    '''
    Define a mask to sellect values at corresponding day-of-year including adjacent variables within the window.
    xr.rolling mean is not employed here for the better performance (by avoiding mapping)
    '''
    doy = ds.doy    
    half = window // 2
    doys = xr.DataArray(np.arange(1, 366, dtype='int16'), dims='doy')

    # circular distance in [-182, 182]
    dist = ((doy - doys + 183) % 365) - 182        # dims: (time, doy)

    # mask samples that fall inside each doy window
    mask = abs(dist) <= half                       # (time, doy)

    ds2 = ds.broadcast_like(mask)
    ds_masked= ds2.where(mask).assign_coords(doy=doys)
    return ds_masked

def clim(ds, window=25):
    if 'doy' not in ds.coords:
        ds = _add_doy_coord(ds)
        
    doy = ds.doy
    ds_masked = _adjacent_doy_sel(ds, window=window)
        
    clim = ds_masked.mean(dim='time')
    return ds.groupby('doy') - clim, clim

def normalize_xr(da):
    t = da.time
    pf = da.polyfit(dim='time', deg=1, skipna=True)
    trend = xr.polyval(t, pf.polyfit_coefficients)
    return da - trend - da.mean('time')

# mhw_threshold --------------------------------------------------------------------------------
def mhw_thresh(ds, mhw_window=25, q=0.9):
    '''
    Marine heatwave threshold is defined as top 90% within the window at each doy of year
    '''
    if 'doy' not in ds.coords:
        ds = _add_doy_coord(ds)
        
    doy = ds.doy
    ds_masked = _adjacent_doy_sel(ds, window=window)
        
    thr = ds_masked.quantile(dim='time',q=q)
    
    return thr
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

def _mhw_event_1d(exceed_bool, min_len=5, max_gap=2):
    '''
    Order: keep ≥min_len runs first, then fill short gaps. Return len==time.
    '''
    a = np.asarray(exceed_bool, dtype=bool)
    labels, n = ndimage.label(a)          # label 1-runs in exceedances
    if n == 0:
        return a
    lengths = np.bincount(labels.ravel())[1:]   # per-run lengths (1..n)
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = lengths >= min_len
    kept = keep[labels]                         # <-- shape == (time,)
    kept_filled = _fill_short_gaps(kept, max_gap=max_gap)
    return kept_filled                               # <-- shape == (time,)

def _fill_short_gaps(sample_cond, max_gap=2):
    '''
    Fill a 0-run with 1s if it is between two 1-runs and its length <= max_gap.
    '''
    a = np.asarray(sample_cond, dtype=bool)
    if a.size == 0:
        return a
    labels, n0 = ndimage.label(~a)  # label 0-runs
    out = a.copy()
    for lab in range(1, n0 + 1):
        idx = np.flatnonzero(labels == lab)
        L, R = idx[0], idx[-1]
        gap_len = R - L + 1
        left_is_one  = (L - 1) >= 0       and out[L - 1]
        right_is_one = (R + 1) < out.size and out[R + 1]
        if left_is_one and right_is_one and gap_len <= max_gap:
            out[L:R+1] = True
    return out


# main API: calc mhw -------------------------------------------------------------------------- 
def mhw_event(da: xr.DataArray, q=0.9, thresh_window=25, min_len=5, max_gap=2) -> xr.Dataset:
    '''
    Return binary (bool-type) file have same structure as the input sea temperature dataset.
    '''
    if 'doy' not in da.coords:
        da = _add_doy_coord(da)

    # Calculate marine heatwave threshold and intensity
    thr = mhw_thresh(da, mhw_window=thresh_window, q=q)
    int = da.groupby('doy') - thr
    
    exceed = int > 0
    
    mhw = xr.apply_ufunc(
        _mhw_event_1d,
        exceed,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[bool],
        dask_gufunc_kwargs={
            'allow_rechunk': True,
            'output_sizes': {'time': da.sizes['time']}
        },
        kwargs={'min_len': min_len, 'max_gap': max_gap},
    )
    mhw = mhw.rename('event')
    int = int.where(mhw).rename('intensity')

    mhw_result = xr.merge([mhw,int]).compute()
    
    return mhw_result
