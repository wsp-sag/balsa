============
IO Utilities
============

Routines for reading and writing demand modelling files, particularly matrices and TMGToolbox network packages.

Matrices
--------

Common
~~~~~~

.. automodule:: balsa.routines.io.common
   :members:

Inro (Emme) format
~~~~~~~~~~~~~~~~~~

.. automodule:: balsa.routines.io.inro
   :members:


OMX format
~~~~~~~~~~

The Python library openmatrix already exists, but doesn't provide a lot of interoperability with Pandas. Balsa provides
wrapper functions that produce Pandas DataFrames and Series directly from OMX files.

.. automodule:: balsa.routines.io.omx
   :members:

Fortran format
~~~~~~~~~~~~~~

.. automodule:: balsa.routines.io.fortran
   :members:

Network Packages (NWP)
----------------------

For more information on the TMGToolbox Network Package format, please visit https://tmg.utoronto.ca/doc/1.6/tmgtoolbox/input_output/ExportNetworkPackage.html

.. automodule:: balsa.routines.io.nwp
   :members:
