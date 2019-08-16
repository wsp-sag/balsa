# Balsa

Balsa is a collection of functions and tools for Python to facilitate travel demand 
forecasting applications and analyses. It is designed to work the the “scientific 
stack” of Python, namely NumPy, Pandas, and Matplotlib; which are optimized for speed
and usability. Most of balsa consists of standalone functions - for input/output, for
analysis, etc. - as well as a few lightweight class-based data structures for specific
applications.

Balsa is published by the Systems Analytics for Policy group inside WSP Canada.

## Key features

 + I/O routines to convert from binary matrix formats (INRO, OMX, and more) to
 Pandas DataFrames and Series.
 + Matrix operations such as balancing, dis/aggregation, and bucket rounding.
 + Plotting functions such a Trip Length Frequency Distributions
 + Pretty Logging utilities for use in program applications
 + Management of JSON configuration files, including comments.
 + and more!  

## Installation

As a private package, Balsa **is not hosted on PyPI or other services that do not
permit private code**. Currently the best way to install Balsa is using `pip` to
install directly from GitHub:

`pip install git+https://github.com/wsp-sag/balsa.git`

Git will prompt you to login to your account (also works with 2FA) before installing.
This requires you to download and install a 
[standalone Git client](https://git-scm.com/downloads) to communicate with GitHub.

**Windows Users:** It is recommended to install Balsa from inside an activated Conda
environment. Balsa uses several packages (NumPy, Pandas, etc.) that will otherwise 
not install correctly from `pip` otherwise. For example:

```
C:\> conda activate base

(base) C:\> pip install git+https://github.com/wsp-sag/balsa.git
...
``` 

## Documentation

HTML documentation is available upon request - until we can find a suitable hosting
service. Just email peter.kucirek@wsp.com to request. 

