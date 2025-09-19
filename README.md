# MarineHeatWaves_xarray

Optimized functions for Marine Heat Wave (MHW) calculations with multi-dimensional datasets using xarray and dask for distributed computing.

This package provides an optimized implementation of the Marine Heat Wave definition of Hobday et al. (2016) that enables distributed computing via the dask framework. It significantly decreases computational cost and makes it easy to work with multi-dimensional, large netCDF-like datasets.

## Features

- **Distributed Computing**: Built on dask for efficient parallel processing
- **Multi-dimensional Support**: Handle datasets with spatial dimensions (lat, lon) using xarray
- **Memory Efficient**: Optimized for large datasets that don't fit in memory
- **Threshold Calculation**: Fast computation of climatological percentile thresholds
- **Event Detection**: Efficient masking and duration calculation
- **Category Classification**: Automatic categorization (moderate, strong, severe, extreme)
- **Compatible Interface**: Similar API to the original marineHeatWaves package

## Installation

```bash
pip install marineheatwaves-xarray
```

Or for development:

```bash
git clone https://github.com/ChaehyeongLee/MarineHeatWaves_xarray.git
cd MarineHeatWaves_xarray
pip install -e .
```

## Quick Start

```python
import xarray as xr
import marineheatwaves_xarray as mhw

# Load your sea surface temperature data
sst = xr.open_dataset('sst_data.nc').sst

# Detect marine heat waves
mhw_events, climatology = mhw.detect(
    sst,
    climatology_period=(1982, 2012),
    percentile=90.0,
    min_duration=5
)

# Calculate block averages
block_stats = mhw.blockAverage(sst.time, mhw_events, block_length=1)

# Calculate trends
mean_vals, trends, trend_errors = mhw.meanTrend(block_stats)
```

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- xarray >= 0.20.0
- dask[array] >= 2021.6.0
- pandas >= 1.3.0
- netcdf4 >= 1.5.0

## Reference

Based on the marine heat wave definition from:

Hobday, A.J. et al. (2016), A hierarchical approach to defining marine heatwaves, Progress in Oceanography, 141, pp. 227-238, doi: 10.1016/j.pocean.2015.12.014

Original marineHeatWaves package: https://github.com/ecjoliver/marineHeatWaves

## License

MIT License - see LICENSE file for details.
