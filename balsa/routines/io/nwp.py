import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pathlib import Path
import re
from typing import Hashable, List, Tuple, Union
import zipfile

EMME_ENG_UNITS = {
    'p': 1E-12,
    'n': 1E-9,
    'u': 1E-6,
    'm': 0.001,
    'k': 1000.0,
    'M': 1E6,
    'G': 1E9,
    'T': 1E12
}


def process_emme_eng_notation_series(s: pd.Series, *, to_dtype=float) -> pd.Series:  # TODO: create generic version...
    """A function to convert Pandas Series containing values in Emme's engineering notation"""
    values = s.str.replace('\D+', '.', regex=True).astype(to_dtype)
    units = s.str.replace('[\d,.]+', '', regex=True).map(EMME_ENG_UNITS).fillna(1.0)
    return values * units


def read_nwp_base_network(nwp_fp: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function to read the base network from a Network Package file (exported from Emme using the TMG Toolbox) into
    DataFrames.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of DataFrames containing the nodes and links
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    header_nodes, header_links, last_line = None, None, None
    with zipfile.ZipFile(nwp_fp) as zf:
        for i, line in enumerate(zf.open('base.211'), start=1):
            line = line.strip().decode('utf-8')
            if line.startswith('c'):
                continue  # Skip comment lines
            if line.startswith('t nodes'):
                header_nodes = i
            elif line.startswith('t links'):
                header_links = i
        last_line = i

        # Read nodes
        n_rows = header_links - header_nodes - 2
        data_types = {
            'c': str, 'Node': np.int64, 'X-coord': float, 'Y-coord': float, 'Data1': float, 'Data2': float,
            'Data3': float, 'Label': str
        }
        nodes = pd.read_csv(zf.open('base.211'), index_col='Node', dtype=data_types, skiprows=header_nodes,
                            nrows=n_rows, delim_whitespace=True)
        nodes.columns = nodes.columns.str.lower()
        nodes.columns = nodes.columns.str.strip()
        nodes.index.name = 'node'
        nodes.rename(columns={'x-coord': 'x', 'y-coord': 'y'}, inplace=True)
        nodes['is_centroid'] = nodes['c'] == 'a*'
        nodes.drop('c', axis=1, inplace=True)

        # Read links
        n_rows = last_line - header_links - 1
        links = pd.read_csv(zf.open('base.211'), index_col=['From', 'To'], skiprows=header_links, nrows=n_rows,
                            delim_whitespace=True, low_memory=False)
        links.columns = links.columns.str.lower()
        links.columns = links.columns.str.strip()
        links.index.names = ['inode', 'jnode']
        mask_mod = links['c'] == 'm'
        n_modified_links = len(links[mask_mod])
        if n_modified_links > 0:
            print(f'Ignored {n_modified_links} modification records in the links table')
        links = links[~mask_mod].drop('c', axis=1)
        if 'typ' in links.columns:
            links.rename(columns={'typ': 'type'}, inplace=True)
        if 'lan' in links.columns:
            links.rename(columns={'lan': 'lanes'}, inplace=True)

        # Data type conversion
        links = links.astype({'modes': str, 'type': int, 'lanes': int, 'vdf': int})  # simple type casting for non-float
        for col in ['length', 'data1', 'data2', 'data3']:
            if is_string_dtype(links[col]):  # these columns are usually string if values use Emme engineering notation
                links[col] = process_emme_eng_notation_series(links[col])
            else:
                links[col] = links[col].astype(float)

    return nodes, links


def read_nwp_exatts_list(nwp_fp: Union[str, Path], **kwargs) -> pd.DataFrame:
    """A function to read the extra attributes present in a Network Package file (exported from Emme using the TMG
    Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.
        **kwargs: Any valid keyword arguments used by ``pandas.read_csv()``.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    kwargs['index_col'] = False
    if 'quotechar' not in kwargs:
        kwargs['quotechar'] = "'"

    with zipfile.ZipFile(nwp_fp) as zf:
        df = pd.read_csv(zf.open('exatts.241'), **kwargs)
        df.columns = df.columns.str.strip()
        df['type'] = df['type'].astype('category')

    return df


def read_nwp_link_attributes(nwp_fp: Union[str, Path], *, attributes: Union[str, List[str]] = None) -> pd.DataFrame:
    """A function to read link attributes from a Network Package file (exported from Emme using the TMG Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.
        attributes (Union[str, List[str]], optional): Defaults to ``None``. Names of link attributes to extract. Note
            that ``'inode'`` and ``'jnode'`` will be included by default.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    if attributes is not None:
        if isinstance(attributes, Hashable):
            attributes = [attributes]
        elif isinstance(attributes, list):
            pass
        else:
            raise RuntimeError

    with zipfile.ZipFile(nwp_fp) as zf:
        df = pd.read_csv(zf.open('exatt_links.241'))
        df.columns = df.columns.str.strip()
        df.set_index(['inode', 'jnode'], inplace=True)

    if attributes is not None:
        df = df[attributes].copy()

    return df


def read_nwp_traffic_results(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read the traffic assignment results from a Network Package file (exported from Emme using the TMG
    Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    with zipfile.ZipFile(nwp_fp) as zf:
        df = pd.read_csv(zf.open('link_results.csv'), index_col=['i', 'j'])
        df.index.names = ['inode', 'jnode']

    return df


def read_nwp_traffic_results_at_countpost(nwp_fp: Union[str, Path], countpost_att: str) -> pd.DataFrame:
    """A function to read the traffic assignment results at countposts from a Network Package file (exported from Emme
    using the TMG Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.
        countpost_att (str): The name of the extra link attribute containing countpost identifiers. Results will be
            filtered using this attribute.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    if not countpost_att.startswith('@'):
        countpost_att = f'@{countpost_att}'

    countpost_links = read_nwp_link_attributes(nwp_fp, attributes=countpost_att)
    countpost_links = countpost_links[countpost_links[countpost_att] > 0]

    results = read_nwp_traffic_results(nwp_fp)
    results = results[results.index.isin(countpost_links.index)].copy()

    return results


def read_nwp_transit_network(nwp_fp: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function to read the transit network from a Network Package file (exported from Emme using the TMG Toolbox)
    into DataFrames.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of DataFrames containing the transt lines and segments.
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    # Parse through transit line transaction file
    seg_cols = ['inode', 'dwt', 'ttf', 'us1', 'us2', 'us3']
    transit_lines = []
    transit_segments = []
    current_tline = None
    with zipfile.ZipFile(nwp_fp) as zf:
        for line in zf.open('transit.221'):
            line = line.strip().decode('utf-8')
            if line.startswith('c') or line.startswith('t') or line.startswith('path'):
                continue  # Skip
            elif line.startswith('a'):
                parts = re.sub(r'\s+', ' ', line.replace("'", ' ')).split(' ')
                parts = parts[1:6] + [' '.join(parts[6:-3])] + parts[-3:]  # reconstruct parts with a joined description
                transit_lines.append(parts)
                current_tline = parts[0]
            else:
                parts = re.sub(r'\s+', ' ', line).split(' ')
                if len(parts) < len(seg_cols):
                    parts = parts + [np.nan] * (len(seg_cols) - len(parts))  # row needed for node... pad with NaNs
                parts.insert(0, current_tline)
                transit_segments.append(parts)

    # Create transit segment dataframe
    transit_segments = pd.DataFrame(transit_segments, columns=['line'] + seg_cols)
    transit_segments['jnode'] = transit_segments.groupby('line')['inode'].shift(-1)
    transit_segments['seg_seq'] = (transit_segments.groupby('line').cumcount() + 1).astype(int)
    transit_segments['loop'] = (transit_segments.groupby(['line', 'inode', 'jnode'])['seg_seq'].cumcount() + 1).astype(int)
    transit_segments.dropna(inplace=True)  # remove rows without dwt, ttf, us1, us2, us3 data (i.e. the padded rows)
    transit_segments = transit_segments[['line', 'inode', 'jnode', 'seg_seq', 'loop', 'dwt', 'ttf', 'us1', 'us2', 'us3']].copy()
    transit_segments['inode'] = transit_segments['inode'].astype(np.int64)
    transit_segments['jnode'] = transit_segments['jnode'].astype(np.int64)
    transit_segments['dwt'] = transit_segments['dwt'].str.replace('dwt=', '')
    transit_segments['ttf'] = transit_segments['ttf'].str.replace('ttf=', '').astype(int)
    transit_segments['us1'] = transit_segments['us1'].str.replace('us1=', '').astype(float)
    transit_segments['us2'] = transit_segments['us2'].str.replace('us2=', '').astype(float)
    transit_segments['us3'] = transit_segments['us3'].str.replace('us3=', '').astype(float)

    # Create transit lines dataframe
    columns = ['line', 'mode', 'veh', 'headway', 'speed', 'description', 'data1', 'data2', 'data3']
    data_types = {  # remember that python 3.6 doesn't guarentee that order is maintained...
        'line': str, 'mode': str, 'veh': int, 'headway': float, 'speed': float, 'description': str, 'data1': float,
        'data2': float, 'data3': float
    }
    transit_lines = pd.DataFrame(transit_lines, columns=columns).astype(data_types)

    return transit_lines, transit_segments


def read_nwp_transit_result_summary(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read and summarize the transit assignment boardings and max volumes from a Network Package file
    (exported from Emme using the TMG Toolbox) by operator and route.

    Note:
        Transit line names in Emme must adhere to the TMG NCS16 for this function to work properly.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    with zipfile.ZipFile(nwp_fp) as zf:
        data_types = {'line': str, 'transit_boardings': float, 'transit_volume': float}
        df = pd.read_csv(zf.open('segment_results.csv'), usecols=data_types.keys(), dtype=data_types)
        df['operator'] = (df['line'].str[:2]).str.replace('\d+', '')
        df['route'] = df['line'].str.replace(r'\D', '').astype(int)
        df = df.groupby(['operator', 'route']).agg({'transit_boardings': 'sum', 'transit_volume': 'max'})
        df.rename(columns={'transit_boardings': 'boardings', 'transit_volume': 'max_volume'}, inplace=True)

    return df


def read_nwp_transit_station_results(nwp_fp: Union[str, Path], station_line_nodes: List[int]) -> pd.DataFrame:
    """A function to read and summarize the transit boardings (on) and alightings (offs) at stations from a Network
    Package file (exported from Emme using the TMG Toolbox).

    Note:
        Ensure that station nodes being specified are on the transit line itself and are not station centroids.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.
        station_line_nodes (List[int]): List of transit line nodes representing transit stops/stations

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    with zipfile.ZipFile(nwp_fp) as zf:
        results = pd.read_csv(zf.open('aux_transit_results.csv'), index_col=['i', 'j'], squeeze=True)

    station_results = pd.DataFrame(index=sorted(station_line_nodes))
    station_results.index.name = 'stn_node'

    mask_on = results.index.get_level_values(1).isin(station_line_nodes)
    station_results['on'] = results.loc[mask_on].groupby(level=1).sum().round(3)

    mask_off = results.index.get_level_values(0).isin(station_line_nodes)
    station_results['off'] = results.loc[mask_off].groupby(level=0).sum().round(3)

    return station_results


def read_nwp_transit_segment_results(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read and summarize the transit segment boardings, alightings, and volumes from a Network Package
    file (exported from Emme using the TMG Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    _, segments = read_nwp_transit_network(nwp_fp)
    segments.set_index(['line', 'inode', 'jnode', 'loop'], inplace=True)

    with zipfile.ZipFile(nwp_fp) as zf:
        results = pd.read_csv(zf.open('segment_results.csv'), index_col=['line', 'i', 'j', 'loop'])

    segments['boardings'] = results['transit_boardings'].round(3)
    segments['volume'] = results['transit_volume'].round(3)
    n_missing_segments = len(segments[segments['boardings'].isnull()])
    if n_missing_segments > 0:
        print(f'Found {n_missing_segments} segments with missing results; their results will be set to 0')
        segments.fillna(0, inplace=True)
    segments.reset_index(inplace=True)

    segments['prev_seg_volume'] = segments.groupby('line')['volume'].shift(1).fillna(0)
    segments['alightings'] = segments.eval('prev_seg_volume + boardings - volume').round(3)

    segments.drop(['dwt', 'ttf', 'us1', 'us2', 'us3', 'prev_seg_volume'], axis=1, inplace=True)
    segments = segments[['line', 'inode', 'jnode', 'seg_seq', 'loop', 'boardings', 'alightings', 'volume']].copy()

    return segments
