import pandas as pd
from astroquery.utils.tap.core import TapPlus  # library to query astronomical databases, TAP services


def safe_to_numeric(col):
    try:
        return pd.to_numeric(col, errors="raise")
    except Exception:
        return col  # leave unchanged if it can't be fully parsed

def load_TAP_data(URL):
    # Connect to the TAP service
    tap = TapPlus(url=URL)

    adql = """
    SELECT TOP 1000
        z.*,
        p.*
    FROM BestDR9.ZooSpec AS z 
    JOIN BestDR7.PhotoObj AS p
    ON p.objid = z.dr7objid
    """

    # Run query
    job = tap.launch_job(adql)
    results = job.get_results()
    df = results.to_pandas()  # convert to pandas DataFrame

    # Convert columns safely to numeric and set index
    df = df.apply(safe_to_numeric).set_index("dr7objid")

    return df


    

def clean_data(df):
    # Apply the mask to filter the DataFrame    
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
        ((df["p_cs_debiased"] >= 0.9) | (df["p_el_debiased"] >= 0.9))
        &
        # medium size
        (df["petroR90_r"] * 2 * 1.5 / 0.4 < 64)
        & (df["petroR90_r"] * 2 / 0.4 > 20)
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

    df_filtered = df[mask][cols_to_keep]

    return df_filtered