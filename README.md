# MarineHeatWaves_xarray

A Python library for detecting and analyzing Marine Heat Waves (MHWs) using xarray and optimized for multi-dimensional, distributed computing-based calculations.

## Overview

This package provides efficient functions for calculating marine heatwave properties from sea surface temperature data. It is designed to work with n-dimensional datasets (e.g., time, latitude, longitude, depth coordinates) and leverages xarray for handling labeled multi-dimensional arrays with optimized chunking for distributed computing.

## Features

- **Multi-dimensional MHW detection**: Works with datasets of any dimensionality (time, lat, lon, depth, etc.)
- **Distributed computing optimized**: Designed for efficient processing with Dask
- **Flexible threshold calculation**: Configurable quantile-based thresholds with sliding window approach
- **Gap filling capabilities**: Intelligent gap filling for short interruptions in heat wave events
- **Climatology computation**: Calculate long-term climatologies with customizable windows
- **Detrending utilities**: Remove linear trends and normalize data

## Installation

### Dependencies

This package requires the following Python libraries:

```bash
pip install numpy pandas xarray scipy
```

For optimal performance with large datasets, also install:

```bash
pip install dask netcdf4 zarr
```

### Installation

Clone this repository and import the module:

```bash
git clone https://github.com/ChaehyeongLee/MarineHeatWaves_xarray.git
cd MarineHeatWaves_xarray
```

## Quick Start

```python
import xarray as xr
import marineheatwaves_xr as mhw

# Load your sea surface temperature data
sst_data = xr.open_dataset('your_sst_file.nc')['sst']

# Detect marine heatwave events
mhw_events = mhw.mhw_event(sst_data, q=0.9, mhw_window=25, min_len=5, max_gap=2)

# Calculate climatology
anomalies, climatology = mhw.clim(sst_data, window=25)

# Calculate thresholds for heatwave detection
thresholds = mhw.mhw_thresh(sst_data, mhw_window=25, q=0.9)
```

## API Reference

### Main Functions

#### `mhw_event(da, q=0.9, mhw_window=25, min_len=5, max_gap=2)`

Detect marine heatwave events in sea surface temperature data.

**Parameters:**
- `da` (xarray.DataArray): Sea surface temperature data with time dimension
- `q` (float, default=0.9): Quantile threshold for defining heatwaves (0.9 = 90th percentile)
- `mhw_window` (int, default=25): Window size in days for calculating thresholds
- `min_len` (int, default=5): Minimum duration in days for a valid heatwave event
- `max_gap` (int, default=2): Maximum gap in days to fill within heatwave events

**Returns:**
- `xarray.DataArray`: Boolean array with same structure as input, True where heatwaves occur

**Example:**
```python
import xarray as xr
import numpy as np
import marineheatwaves_xr as mhw

# Create sample data
time = pd.date_range('2000-01-01', '2020-12-31', freq='D')
sst = xr.DataArray(
    np.random.randn(len(time), 10, 10) + 20,  # Random SST around 20°C
    dims=['time', 'lat', 'lon'],
    coords={'time': time, 'lat': range(10), 'lon': range(10)}
)

# Detect heatwaves
heatwaves = mhw.mhw_event(sst, q=0.9, min_len=5)
print(f"Heatwave frequency: {heatwaves.sum().values} days")
```

#### `mhw_thresh(ds, mhw_window=25, q=0.9)`

Calculate marine heatwave threshold values for each day of year.

**Parameters:**
- `ds` (xarray.DataArray): Sea surface temperature data
- `mhw_window` (int, default=25): Window size for threshold calculation
- `q` (float, default=0.9): Quantile for threshold (0.9 = 90th percentile)

**Returns:**
- `xarray.DataArray`: Threshold values for each day of year

#### `clim(ds, window=25)`

Calculate climatology and anomalies from the input dataset.

**Parameters:**
- `ds` (xarray.DataArray): Input temperature data
- `window` (int, default=25): Window size in days for climatology calculation

**Returns:**
- `tuple`: (anomalies, climatology) as xarray.DataArrays

**Example:**
```python
# Calculate climatology and anomalies
anomalies, climatology = mhw.clim(sst, window=25)

# Plot climatology for a specific location
climatology.isel(lat=5, lon=5).plot()
```

#### `normalize_xr(da)`

Remove linear trends and mean from the data.

**Parameters:**
- `da` (xarray.DataArray): Input data with time dimension

**Returns:**
- `xarray.DataArray`: Detrended and normalized data

### Utility Functions

#### `_fill_missing(ds)`
Fill missing values in time coordinate and remove February 29th.

#### `_add_doy_coord(ds)`
Add day-of-year coordinate to dataset.

#### `_adjacent_doy_sel(ds, window=25)`
Select adjacent day-of-year values within specified window.

## Performance Optimization

For optimal performance with large datasets:

### Chunking Strategy

```python
# Recommended chunking approach
sst_chunked = sst.chunk({'time': -1, 'lat': 50, 'lon': 50})

# Adjust chunk sizes based on available memory
# Rule of thumb: chunk size should be 1/10 to 1/15 of available worker memory
```

### Memory Management

- Set the largest available "time" coordinate first when chunking
- Use `.persist()` for intermediate results that will be reused
- Consider using Zarr format for large datasets

### Dask Configuration

```python
import dask
from dask.distributed import Client

# Configure Dask for distributed computing
client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')

# Process data
with dask.config.set(scheduler='threads'):
    result = mhw.mhw_event(sst_data)
```

## Examples

### Example 1: Basic Heatwave Detection

```python
import xarray as xr
import pandas as pd
import numpy as np
import marineheatwaves_xr as mhw

# Load or create SST data
sst = xr.tutorial.open_dataset("rasm").Tair.isel(time=slice(0, 365))

# Detect marine heatwaves
mhw_events = mhw.mhw_event(sst, q=0.9, min_len=5, max_gap=2)

# Count heatwave days for each location
heatwave_days = mhw_events.sum(dim='time')
print(f"Maximum heatwave days at any location: {heatwave_days.max().values}")
```

### Example 2: Custom Threshold Analysis

```python
# Calculate different threshold levels
thresh_90 = mhw.mhw_thresh(sst, q=0.9)  # 90th percentile
thresh_95 = mhw.mhw_thresh(sst, q=0.95) # 95th percentile

# Compare thresholds
threshold_diff = thresh_95 - thresh_90
threshold_diff.mean().plot(title='Difference between 95th and 90th percentile thresholds')
```

### Example 3: Climatology Analysis

```python
# Calculate climatology with different window sizes
anom_25, clim_25 = mhw.clim(sst, window=25)
anom_11, clim_11 = mhw.clim(sst, window=11)

# Compare climatologies
clim_diff = clim_25 - clim_11
clim_diff.plot(title='Difference in climatology (25-day vs 11-day window)')
```

## Testing

The package has been tested with synthetic data to ensure basic functionality works correctly. To run a simple test:

```python
import marineheatwaves_xr as mhw
import xarray as xr
import numpy as np
import pandas as pd

# Create test data
time = pd.date_range('2000-01-01', '2001-12-31', freq='D')
sst = xr.DataArray(
    np.random.randn(len(time), 5, 5) + 20,
    dims=['time', 'lat', 'lon'],
    coords={'time': time, 'lat': range(5), 'lon': range(5)}
)

# Test functionality
anomalies, climatology = mhw.clim(sst, window=25)
thresholds = mhw.mhw_thresh(sst, q=0.9)
heatwaves = mhw.mhw_event(sst, q=0.9, min_len=5)

print("All functions working correctly!")
```

## Data Requirements

The input sea surface temperature data should:

- Be an xarray.DataArray with a time dimension
- Have daily or sub-daily temporal resolution
- Include coordinates for spatial dimensions (lat, lon, etc.)
- Cover multiple years for meaningful climatology calculation

## Algorithm Details

The marine heatwave detection follows these steps:

1. **Data preprocessing**: Fill missing values and handle leap days
2. **Day-of-year coordinate**: Add DOY coordinate for seasonal analysis
3. **Threshold calculation**: Calculate quantile-based thresholds using sliding window
4. **Exceedance detection**: Identify when temperatures exceed thresholds
5. **Event filtering**: Apply minimum duration and gap-filling criteria
6. **Output generation**: Return boolean mask indicating heatwave periods

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```
Lee, C. (2025). MarineHeatWaves_xarray: Optimized functions for multi-dimensional, 
distributed computing-based MHW calculations. GitHub repository: 
https://github.com/ChaehyeongLee/MarineHeatWaves_xarray
```

## References

- Hobday, A. J., et al. (2016). A hierarchical approach to defining marine heatwaves. Progress in Oceanography, 141, 227-238.
- Oliver, E. C., et al. (2018). Marine heatwaves off eastern Tasmania: Trends, interannual variability, and predictability. Progress in Oceanography, 161, 116-130.
