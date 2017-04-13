# balsa
Collection of modelling utilities. Works with both Python 2.7 and Python 3.5+

The current documentation can be found [on the Wiki page](https://github.com/pbsag/balsa/wiki). Some components are under construction.

## Contents:

* __cheval__: High-performance engine for evaluating discrete-choice (logit) models over DataFrames where utilities can be specified as expressions. Works with multinomial or nested models. Also includes the LinkedDataFrame class, a subclass of Pandas DataFrame which can be linked to other data frames.
* __matrices__: Matrix balancing, as well as I/O for binary matrices.
* __pandas_utils__: Utilities (such as `fast_stack`, and `align_cateogires`) for the Pandas library
* __configuration__: Parsing and validation of JSON-based configuration files
* __scribe__: Convenient functions for logging model information during a run.
