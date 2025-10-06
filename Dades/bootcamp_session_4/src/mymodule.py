import urllib

import pandas as pd
from astroquery.utils.tap.core import (
    TapPlus,
)


def safe_to_numeric(col: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric if all values are parseable.

    Attempts to parse the entire column as numbers. If any value fails, the
    original column is returned unchanged (strings remain strings).

    Args:
        col: Input pandas Series.

    Returns:
        A Series with numeric dtype (int/float) if fully parseable;
        otherwise the original Series.

    Examples:
        >>> import pandas as pd
        >>> safe_to_numeric(pd.Series(["1","2","3"])).dtype
        dtype('int64')
        >>> safe_to_numeric(pd.Series(["1","x"])).dtype  # unchanged
        dtype('O')
    """
    try:
        return pd.to_numeric(col, errors="raise")

    except Exception:
        return col  # leave unchanged if it can't be fully parsed


def query_tap(service: str, adql: str) -> pd.DataFrame:
    """Run an ADQL query against a TAP service and return a typed DataFrame.

    Connects to the TAP endpoint with Astroquery's TapPlus, executes the query,
    converts the result to a pandas DataFrame, and applies ``safe_to_numeric``
    column-wise to coerce truly-numeric columns.

    Args:
        service: TAP service URL (e.g. ``https://datalab.noirlab.edu/tap``).
        adql: ADQL query string.

    Returns:
        DataFrame with columns converted to numeric where safe.

    Raises:
        Exception: Propagates connection/query errors from TapPlus.

    Examples:
        >>> url = "https://datalab.noirlab.edu/tap"
        >>> _ = isinstance(adql := "SELECT TOP 1 1 AS x", str)
        >>> # query_tap(url, adql)  # would hit network
    """
    # Connect to the TAP service
    tap = TapPlus(url=service)

    # Run the query
    job = tap.launch_job(adql)
    results = job.get_results()
    return results.to_pandas().apply(safe_to_numeric)


def filter_data(
    df: pd.DataFrame, confidence: float = 0.8, size_range: tuple[int, int] = (20, 64)
) -> pd.DataFrame:
    """Filter SDSS-like photometric data with quality and morphology cuts.

    The mask enforces:
      - plausible magnitudes and small photometric errors,
      - high morphology confidence (either class >= ``confidence``),
      - a medium angular size range derived from ``petroR90_r``.

    Args:
        df: Input DataFrame containing SDSS-like columns (see Notes).
        confidence: Minimum debiased probability for either class.
        size_range: (min, max) size limits in pixels for the derived diameter.

    Returns:
        Filtered DataFrame with a curated set of columns.

    Notes:
        Required columns include:
        ``specobjid, objid, ra, dec, p_el_debiased, p_cs_debiased, spiral, elliptical,
        petroR50_r, petroR90_r, modelMag_[ugriz], modelMagErr_[ugriz], extinction_[ugriz]``.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"modelMag_u":[20], "modelMag_g":[19], "modelMag_r":[18],
        ...                    "modelMag_i":[18], "modelMag_z":[17],
        ...                    "modelMagErr_u":[0.4], "modelMagErr_g":[0.04],
        ...                    "modelMagErr_r":[0.04], "modelMagErr_i":[0.04],
        ...                    "modelMagErr_z":[0.09],
        ...                    "p_cs_debiased":[0.9], "p_el_debiased":[0.1],
        ...                    "spiral":[1.0], "elliptical":[0.0],
        ...                    "petroR50_r":[3.0], "petroR90_r":[5.0],
        ...                    "specobjid":[1], "objid":[2], "ra":[180.], "dec":[0.],
        ...                    "extinction_u":[0.1], "extinction_g":[0.1], "extinction_r":[0.1],
        ...                    "extinction_i":[0.1], "extinction_z":[0.1]})
        >>> out = filter_data(df)
        >>> len(out)
        1
    """
    mask = (
        # correct magnitudes
        (df["modelMag_u"] > -30)
        & (df["modelMag_g"] > -30)
        & (df["modelMag_r"] > -30)
        & (df["modelMag_i"] > -30)
        & (df["modelMag_z"] > -30)
        &
        # reasonable errors
        (df["modelMagErr_u"] < 0.5)
        & (df["modelMagErr_g"] < 0.05)
        & (df["modelMagErr_r"] < 0.05)
        & (df["modelMagErr_i"] < 0.05)
        & (df["modelMagErr_z"] < 0.1)
        &
        # very certain about the classification
        ((df["p_cs_debiased"] >= confidence) | (df["p_el_debiased"] >= confidence))
        &
        # medium size
        (df["petroR90_r"] * 2 / 0.4 > size_range[0])
        & (df["petroR90_r"] * 2 * 1.5 / 0.4 < size_range[1])
    )

    cols_to_keep = (
        [
            "specobjid",
            "objid",
            "ra",
            "dec",
            "p_el_debiased",
            "p_cs_debiased",
            "spiral",
            "elliptical",
        ]
        + ["petroR50_r", "petroR90_r"]
        + [f"modelMag_{f}" for f in "ugriz"]
        + [f"extinction_{f}" for f in "ugriz"]
    )

    return df[mask][cols_to_keep]


def retrieve_img(ra: float, dec: float, scale: float, width: int, height: int) -> bytes:
    """Retrieve an SDSS JPEG cutout for a sky position.

    Args:
        ra: Right ascension in degrees (ICRS).
        dec: Declination in degrees (ICRS).
        scale: Arcsec per pixel (SDSS cutout API).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Raw JPEG bytes.

    Raises:
        URLError: If the request fails or the service is unreachable.

    Examples:
        >>> # bytes = retrieve_img(180.0, 0.0, 0.4, 128, 128)  # network call
    """
    URL = (
        "https://skyserver.sdss.org/DR19/SkyserverWS/ImgCutout/getjpeg?"
        "ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}"
    )

    url = URL.format(**locals())
    response = urllib.request.urlopen(url)
    return response.read()
