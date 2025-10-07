import pandas as pd

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