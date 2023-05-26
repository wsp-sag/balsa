from __future__ import annotations

from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd

from ..matrices import fast_unstack

try:
    import openmatrix as omx
except ImportError:
    omx = None

MATRIX_TYPES = Union[pd.DataFrame, pd.Series, np.ndarray]


if omx is not None:
    def read_omx(file: str, *, matrices: Iterable[str] = None, mapping: str = None, raw: bool = False,
                 tall: bool = False, squeeze: bool = True) -> Union[MATRIX_TYPES, Dict[str, MATRIX_TYPES]]:
        """
        Reads Open Matrix (OMX) files. An OMX file can contain multiple matrices, so this function
        typically returns a Dict.

        Args:
            file: OMX file from which to read. Cannot be an open file handler.
            matrices: List of matrices to read from the file. If None, all matrices will be read.
            mapping: The zone number mapping to use, if known in advance. If None, and the OMX file only contains
                one mapping, then that one is used. No mapping is read if raw is False.
            raw: If True, matrices will be returned as raw Numpy arrays. Otherwise, Pandas objects are returned
            tall: If True, matrices will be returned in 1D format (pd.Series if raw is False). Otherwise, a 2D object
                is returned.
            squeeze: If True, and the file contains exactly one matrix, return that matrix instead of a Dict.

        Returns:
            The matrix, or matrices contained in the OMX file.

        """
        file = str(file)
        with omx.open_file(file, mode='r') as omx_file:
            if mapping is None and not raw:
                all_mappings = omx_file.list_mappings()
                assert len(all_mappings) == 1
                mapping = all_mappings[0]

            if matrices is None:
                matrices = sorted(omx_file.list_matrices())
            else:
                matrices = sorted(matrices)

            if not raw:
                labels = pd.Index(omx_file.mapping(mapping).keys())
                if tall:
                    labels = pd.MultiIndex.from_product([labels, labels], names=['o', 'd'])

            return_value = {}
            for matrix_name in matrices:
                wrapper = omx_file[matrix_name]
                matrix = wrapper.read()

                if tall:
                    n = matrix.shape[0] * matrix.shape[1]
                    matrix.shape = n

                if not raw:
                    if tall: matrix = pd.Series(matrix, index=labels)
                    else: matrix = pd.DataFrame(matrix, index=labels, columns=labels)

                    matrix.name = matrix_name

                return_value[matrix_name] = matrix

            if len(matrices) == 1 and squeeze:
                return return_value[matrices[0]]
            return return_value


    def to_omx(file: str, matrices: Dict[str, MATRIX_TYPES], *, zone_index: pd.Index = None, title: str = '',
               descriptions: Dict[str, str] = None, attrs: Dict[str, dict] = None, mapping: str = 'zone_numbers'):
        """Creates a new (or overwrites an old) OMX file with a collection of matrices.

        Args:
            file: OMX to write.
            matrices: Collection of matrices to write. MUST be a dict, to permit the encoding of matrix metadata,
                and must contain the same types: all Numpy arrays, all Series, or all DataFrames. Checking is done to
                ensure that all items have the same shape and labels.
            zone_index: Override zone labels to use. Generally only useful if writing a dict of raw Numpy arrays.
            title: The title saved in the OMX file.
            descriptions: A dict of descriptions (one for each given matrix), or None to not use.
            attrs: A dict of dicts (one for each given matrix), or None to not use
            mapping: Name of the mapping internal to the OMX file
        """

        matrices, zone_index = _prep_matrix_dict(matrices, zone_index)

        if descriptions is None:
            descriptions = {name: '' for name in matrices.keys()}
        if attrs is None:
            attrs = {name: None for name in matrices.keys()}

        file = str(file)  # Converts from Path
        with omx.open_file(file, mode='w', title=title) as omx_file:
            omx_file.create_mapping(mapping, zone_index.tolist())

            for name, array in matrices.items():
                description = descriptions[name]
                attr = attrs[name]

                omx_file.create_matrix(name, obj=np.ascontiguousarray(array), title=description, attrs=attr)

        return


    def _prep_matrix_dict(matrices, desired_zone_index):
        collection_type = _check_types(matrices)

        if collection_type == 'RAW':
            checked, n = _check_raw_matrices(matrices)
            zone_index = pd.Index(range(n))
        elif collection_type == 'SERIES':
            checked, zone_index = _check_matrix_series(matrices)
        elif collection_type == 'FRAME':
            checked, zone_index = _check_matrix_frames(matrices)
        else:
            raise NotImplementedError(collection_type)

        if desired_zone_index is not None:
            assert desired_zone_index.equals(zone_index)

        return checked, zone_index


    def _check_types(matrices):
        gen = iter(matrices.values())
        first = next(gen)

        item_type = 'RAW'
        if isinstance(first, pd.Series):
            item_type = 'SERIES'
        elif isinstance(first, pd.DataFrame):
            item_type = 'FRAME'

        msg = "All items must be the same type"

        for item in gen:
            if item_type == 'FRAME':
                assert isinstance(item, pd.DataFrame), msg
            elif item_type == 'SERIES':
                assert isinstance(item, pd.Series), msg
            else:
                assert isinstance(item, np.ndarray), msg

        return item_type


    def _check_raw_matrices(matrices):
        gen = iter(matrices.items())
        name, matrix = next(gen)

        n_dim = len(matrix.shape)
        if n_dim == 1:
            shape = matrix.shape[0]
            n = int(shape ** 0.5)
            assert n * n == shape, "Only tall matrices that decompose to square shapes are permitted."
            matrix = matrix[...]
            matrix.shape = n, n
        elif n_dim == 2:
            n, cols = matrix.shape
            assert n == cols, "Only square matrices are permitted"
        else:
            raise TypeError("Only 1D and 2D arrays can be saved to OMX files")

        retval = {name: matrix}
        for name, matrix in gen:
            assert len(matrix.shape) == n_dim

            if n_dim == 1:
                assert matrix.shape[0] == (n * n)
                matrix = matrix[...]
                matrix.shape = n, n
            else:
                assert matrix.shape == (n, n)

            retval[name] = matrix

        return retval, n


    def _check_matrix_series(matrices):
        gen = iter(matrices.items())
        name, matrix = next(gen)

        tall_index = matrix.index
        assert tall_index.nlevels == 2
        matrix = matrix.unstack()
        zone_index = matrix.index
        assert zone_index.equals(matrix.columns)

        retval = {name: matrix.values}
        for name, matrix in gen:
            assert tall_index.equals(matrix.index)
            matrix = fast_unstack(matrix, zone_index, zone_index).values
            retval[name] = matrix
        return retval, zone_index


    def _check_matrix_frames(matrices):
        gen = iter(matrices.items())
        name, matrix = next(gen)

        zone_index = matrix.index
        assert zone_index.equals(matrix.columns)

        retval = {name: matrix.values}
        for name, matrix in gen:
            assert zone_index.equals(matrix.index)
            assert zone_index.equals(matrix.columns)
            retval[name] = matrix.values
        return retval, zone_index
else:
    def read_omx(*args, **kwargs):
        raise NotImplementedError()

    def to_omx(*args, **kwargs):
        raise NotImplementedError()
