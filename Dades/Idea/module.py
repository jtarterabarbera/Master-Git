import pandas as pd
from astroquery.hips2fits import hips2fits
import astropy.units as u
from astropy.coordinates import Longitude, Latitude, Angle


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


def get_image(ra: float, dec: float, r: float, width: int, height: int) -> bytes:
    hips = "CDS/P/SDSS9/color"
    result = hips2fits.query(
        hips=hips,
        width=width,
        height=height,
        ra=Longitude(ra * u.deg),
        dec=Latitude(dec * u.deg),
        fov=Angle(r * 2 * u.arcsec),
        projection="AIT",
        get_query_payload=False,
        format="jpg",
    )
    return result
