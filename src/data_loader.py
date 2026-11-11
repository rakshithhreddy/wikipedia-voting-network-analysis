"""Data Loader Module for Wikipedia Voting Network.

Handles automatic downloading of the Stanford SNAP Wiki-Vote dataset,
gzip decompression, parsing, and directed graph construction.
"""

import gzip
import logging
import os
import urllib.request
from typing import Optional, Tuple, Union

import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SNAP_WIKI_VOTE_URL = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def download_and_extract_dataset(
    url: str = SNAP_WIKI_VOTE_URL,
    data_dir: str = DEFAULT_DATA_DIR,
    force_download: bool = False,
) -> str:
    """Download the official SNAP Wiki-Vote dataset and decompress it.

    Parameters
    ----------
    url : str
        Source URL for the dataset archive.
    data_dir : str
        Destination directory to store raw and uncompressed files.
    force_download : bool
        If True, re-downloads even if the file exists.

    Returns
    -------
    str
        Path to the uncompressed txt file.
    """
    os.makedirs(data_dir, exist_ok=True)
    gz_filename = os.path.basename(url)
    gz_path = os.path.join(data_dir, gz_filename)
    txt_path = os.path.join(data_dir, "wiki-Vote.txt")

    # If txt already exists and not forced, return it
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 and not force_download:
        logger.info(f"Using existing extracted dataset at {txt_path}")
        return txt_path

    # Download archive if not present
    if not os.path.exists(gz_path) or os.path.getsize(gz_path) == 0 or force_download:
        logger.info(f"Downloading official SNAP Wiki-Vote dataset from {url}...")
        try:
            # Custom headers for reliable download
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SNA-Research/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(gz_path, "wb") as out_file:
                out_file.write(response.read())
            logger.info(f"Downloaded {os.path.getsize(gz_path)} bytes to {gz_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset from {url}: {e}")
            raise RuntimeError(f"Network error downloading SNAP dataset: {e}") from e

    # Decompress gzip
    logger.info(f"Extracting {gz_path} to {txt_path}...")
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8") as f_in:
            with open(txt_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    f_out.write(line)
        logger.info(f"Successfully extracted {os.path.getsize(txt_path)} bytes.")
    except Exception as e:
        logger.error(f"Failed to decompress {gz_path}: {e}")
        raise RuntimeError(f"Error extracting gzip archive: {e}") from e

    return txt_path


def load_raw_edge_dataframe(file_path: Optional[str] = None) -> pd.DataFrame:
    """Parse the raw Wiki-Vote text file into a clean Pandas DataFrame.

    In the Stanford SNAP Wiki-Vote dataset:
    - Lines starting with '#' are metadata comments.
    - Each line represents: FromNodeId (voter) -> ToNodeId (candidate voted on).

    Parameters
    ----------
    file_path : Optional[str]
        Path to wiki-Vote.txt. If None, checks default data directory and fallbacks.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['source', 'target'] of type int64.
    """
    if file_path is None or not os.path.exists(file_path):
        txt_path = os.path.join(DEFAULT_DATA_DIR, "wiki-Vote.txt")
        gz_path = os.path.join(DEFAULT_DATA_DIR, "wiki-Vote.txt.gz")
        edges_csv = os.path.join(os.path.dirname(DEFAULT_DATA_DIR), "exports", "wiki_vote_edges.csv")

        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            file_path = txt_path
        elif os.path.exists(gz_path) and os.path.getsize(gz_path) > 0:
            file_path = download_and_extract_dataset(data_dir=DEFAULT_DATA_DIR)
        elif os.path.exists(edges_csv) and os.path.getsize(edges_csv) > 0:
            logger.info(f"Loading edge list from pre-packaged export: {edges_csv}")
            df = pd.read_csv(edges_csv)
            # Normalize column names
            df = df.rename(columns={"source_voter": "source", "target_candidate": "target"})
            return df[["source", "target"]].astype("int64")
        else:
            file_path = download_and_extract_dataset(url=SNAP_WIKI_VOTE_URL, data_dir=DEFAULT_DATA_DIR)

    logger.info(f"Loading edge list from {file_path}...")
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        names=["source", "target"],
        dtype={"source": "int64", "target": "int64"},
    )
    logger.info(f"Loaded {len(df):,} directed voting interactions.")
    return df


def build_directed_graph(data_source: Optional[Union[str, pd.DataFrame]] = None) -> nx.DiGraph:
    """Construct a NetworkX directed graph (DiGraph) from Wiki-Vote data.

    Preserves directional semantics:
    An edge (u -> v) means user u voted in support/opposition of candidate v
    for adminship.

    Parameters
    ----------
    data_source : Optional[Union[str, pd.DataFrame]]
        Either a file path to wiki-Vote.txt or an existing DataFrame.

    Returns
    -------
    nx.DiGraph
        Directed graph representing Wikipedia voting network.
    """
    if isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        df = load_raw_edge_dataframe(data_source)

    logger.info("Constructing NetworkX DiGraph...")
    G = nx.DiGraph()
    # Using add_edges_from with zip for peak memory and speed efficiency
    G.add_edges_from(zip(df["source"], df["target"]))

    # Add network title attribute
    G.graph["title"] = "Wikipedia Adminship Voting Network (SNAP Wiki-Vote)"
    G.graph["description"] = (
        "Directed network of Wikipedia voting on promotion to administrator. "
        "Directed edge (u -> v) indicates user u cast a vote on candidate v."
    )

    logger.info(
        f"Constructed DiGraph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} directed edges."
    )
    return G


if __name__ == "__main__":
    txt_file = download_and_extract_dataset()
    graph = build_directed_graph(txt_file)
    print(f"Graph loaded successfully: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
