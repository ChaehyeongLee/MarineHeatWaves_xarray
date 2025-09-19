"""
MarineHeatWaves_xarray

Optimized functions for Marine Heat Wave calculations with multi-dimensional 
datasets using xarray and dask for distributed computing.

This package provides an optimized implementation of the Marine Heat Wave (MHW) 
definition of Hobday et al. (2016) that enables distributed computing via the 
dask framework, significantly decreasing computational cost when dealing with 
multi-dimensional, large netCDF-like datasets.

Based on the original marineHeatWaves package by Eric Oliver:
https://github.com/ecjoliver/marineHeatWaves

Key Features:
- Distributed computing support via dask
- Multi-dimensional dataset handling with xarray
- Optimized threshold calculation
- Event masking and duration calculation
- MHW category determination (moderate, strong, severe, extreme)
- Efficient memory usage for large datasets

Reference:
Hobday, A.J. et al. (2016), A hierarchical approach to defining marine heatwaves, 
Progress in Oceanography, 141, pp. 227-238, doi: 10.1016/j.pocean.2015.12.014
"""

from .core import detect, blockAverage, meanTrend, rank
from .utils import calculate_threshold, calculate_climatology, pad_missing_values
from .version import __version__

__all__ = [
    "detect",
    "blockAverage", 
    "meanTrend",
    "rank",
    "calculate_threshold",
    "calculate_climatology",
    "pad_missing_values",
    "__version__",
]