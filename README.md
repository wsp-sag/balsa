# Balsa (wsp-balsa)

Balsa is a collection of functions and tools for Python to facilitate travel demand forecasting applications and analyses. It is designed to work the the “scientific stack” of Python, namely NumPy, Pandas, and Matplotlib; which are optimized for speed and usability. Most of balsa consists of standalone functions - for input/output, for analysis, etc. - as well as a few lightweight class-based data structures for specific applications.

Balsa is owned and published by WSP Canada's Systems Analytics for Policy group.

## Key features

- I/O routines to convert from binary matrix formats (INRO, OMX, and more) to Pandas DataFrames and Series.
- Matrix operations such as balancing, dis/aggregation, and bucket rounding.
- Plotting functions such a Trip Length Frequency Distributions
- Pretty Logging utilities for use in program applications
- Management of JSON configuration files, including comments.
- and more!

Balsa is compatible with Python 2.7 and 3.5+

## Installation

### With `pip`

As a private package, Balsa **is not hosted on PyPI or other services that do not permit private code**. Currently the best way to install Balsa is using `pip` to install directly from GitHub:

```batch
pip install git+https://github.com/wsp-sag/balsa.git
```

Git will prompt you to login to your account (also works with 2FA) before installing. This requires you to download and install a [standalone Git client](https://git-scm.com/downloads) to communicate with GitHub.

> **Windows Users:** It is recommended to install Balsa from inside an activated Conda environment. Balsa uses several packages (NumPy, Pandas, etc.) that will otherwise not install correctly from `pip` otherwise. For example:

```batch
C:\> conda activate base

(base) C:\> pip install git+https://github.com/wsp-sag/balsa.git
...
```

### With `conda`

Balsa can be installed with Conda, but requires you to install it from a local Conda channel. This can be done by using [conda-build](https://github.com/conda/conda-build), which will create a Conda package for Balsa (that has been cloned from GitHub onto your machine) and set up a local Conda channel (i.e. `conda-bld`) in your Conda installation folder. conda-build must be installed in your base Conda environment. Once the Conda package is built, you can install it to your Conda environment of choice using `conda install`.

The following code block provides the commands to install Balsa using Conda.

```batch
(base) C:\> conda build "<path to local balsa repository folder>/conda_recipe"

...

(base) C:\> conda install -c "<path to your conda installation folder>/conda-bld" wsp-balsa
```

## Documentation

HTML documentation is available upon request. 
