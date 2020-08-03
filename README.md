# Balsa (wsp-balsa)

[![Conda Latest Release](https://anaconda.org/wsp_sap/wsp-balsa/badges/version.svg)](https://anaconda.org/wsp_sap/wsp-balsa)
[![Conda Last Updated](https://anaconda.org/wsp_sap/wsp-balsa/badges/latest_release_date.svg)](https://anaconda.org/wsp_sap/wsp-balsa)
[![Platforms](https://anaconda.org/wsp_sap/wsp-balsa/badges/platforms.svg)](https://anaconda.org/wsp_sap/wsp-balsa)
[![License](https://anaconda.org/wsp_sap/wsp-balsa/badges/license.svg)](https://github.com/wsp-sag/balsa/blob/master/LICENSE)

Balsa is a collection of functions and tools for Python to facilitate travel demand forecasting applications and analyses. It is designed to work the the “scientific stack” of Python, namely NumPy, Pandas, and Matplotlib; which are optimized for speed and usability. Most of balsa consists of standalone functions - for input/output, for analysis, etc. - as well as a few lightweight class-based data structures for specific applications.

Balsa is owned and published by WSP Canada's Systems Analytics for Policy group.

## Key features

- I/O routines to convert from binary matrix formats (INRO, OMX, and more) to Pandas DataFrames and Series.
- Matrix operations such as balancing, dis/aggregation, and bucket rounding.
- Plotting functions such a Trip Length Frequency Distributions
- Pretty Logging utilities for use in program applications
- Management of JSON configuration files, including comments.
- and more!

Balsa is compatible with Python 3.5+

## Installation

> **For TRESO users:** TRESO is only compatible with the [`v0.6.1`](https://github.com/wsp-sag/balsa/releases/tag/v0.6.1) release of Balsa, which can only be installed directly from GitHub using `pip`.

### With `conda`

Balsa can be installed with conda by running the following command:

```batch
conda install -c wsp_sap wsp-balsa
```

### With `pip`

Balsa can be installed directly from GitHub using `pip` by running the following command:

```batch
pip install git+https://github.com/wsp-sag/balsa.git
```

> **Windows Users:** It is recommended to install Balsa from inside an activated Conda environment. Balsa uses several packages (NumPy, Pandas, etc.) that will otherwise not install correctly from `pip` otherwise. For example:

```batch
C:\> conda activate base

(base) C:\> pip install git+https://github.com/wsp-sag/balsa.git
...
```

## Documentation

HTML documentation is available upon request.
