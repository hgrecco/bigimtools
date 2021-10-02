.. image:: https://img.shields.io/pypi/v/bigimtools.svg
    :target: https://pypi.python.org/pypi/bigimtools
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/l/bigimtools.svg
    :target: https://pypi.python.org/pypi/bigimtools
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/bigimtools.svg
    :target: https://pypi.python.org/pypi/bigimtools
    :alt: Python Versions

.. image:: https://github.com/hgrecco/bigimtools/workflows/CI/badge.svg?branch=main
    :target: https://github.com/hgrecco/bigimtools/actions?query=workflow%3ACI

.. image:: https://github.com/hgrecco/bigimtools/workflows/Lint/badge.svg?branch=main
    :target: https://github.com/hgrecco/bigimtools/actions?query=workflow%3ALint

.. image:: https://coveralls.io/repos/github/hgrecco/bigimtools/badge.svg?branch=main
    :target: https://coveralls.io/github/hgrecco/bigimtools?branch=main

.. image:: https://readthedocs.org/projects/bigimtools/badge/
    :target: http://bigimtools.readthedocs.org/
    :alt: Docs


bigimtools: manipulate large images
===================================

bigimtools is a Python package to handle large images.

It currently provides a TiledImage class that handles
large images split into tiles with or without overlap.
Those images can be stored in different backends: HDF5 files,
images in a folder, numpy arrays or in memory dictionaries.

A few utilities are provided:

- tiler module: functions to split image into tiles, join them,
  equalize them and more
- dzi module: functions to create a deep zoom image.


Quick Installation
------------------

To install bigimtools, simply (*soon*):

.. code-block:: bash

    $ pip install bigimtools

or utilizing conda, with the conda-forge channel (*soon*):

.. code-block:: bash

    $ conda install -c conda-forge bigimtools

and then simply enjoy it!


----

bigimtools is maintained by a community. See AUTHORS_ for a complete list.

To review an ordered list of notable changes for each version of a project,
see CHANGES_


.. _`NumPy`: http://www.numpy.org/
.. _`PILLOW`: https://pillow.readthedocs.io/en/stable/
.. _`pytest`: https://docs.pytest.org/
.. _`H5PY`: https://www.h5py.org/
.. _`AUTHORS`: https://github.com/hgrecco/bigimtools/blob/master/AUTHORS
.. _`CHANGES`: https://github.com/hgrecco/bigimtools/blob/master/CHANGES
