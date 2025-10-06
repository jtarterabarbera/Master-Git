import pandas as pd

import mymodule

TAP_URL = "http://tap.roe.ac.uk/ssa"


def test_safe_to_numeric_all_numeric_ints():
    s = pd.Series(["1", "2", "3"])
    out = mymodule.safe_to_numeric(s)
    assert pd.api.types.is_integer_dtype(out), f"Expected int dtype, got {out.dtype}"
    assert list(out) == [1, 2, 3]


def test_safe_to_numeric_mixed_kept_as_strings():
    s = pd.Series(["1", "x", "3"])
    out = mymodule.safe_to_numeric(s)
    # Should remain unchanged (strings) if any parse error
    assert out.equals(s)
    assert out.dtype == object


def test_retrieve_adql():
    adql = """
    SELECT TOP 1
        z.*,
        p.*
    FROM BestDR9.ZooSpec AS z
    JOIN BestDR7.PhotoObj AS p
    ON p.objid = z.dr7objid
    """
    df = mymodule.query_tap(TAP_URL, adql).set_index("dr7objid")
    assert len(df) == 1


def test_filter_data():
    adql = """
    SELECT TOP 10
        z.*,
        p.*
    FROM BestDR9.ZooSpec AS z
    JOIN BestDR7.PhotoObj AS p
    ON p.objid = z.dr7objid
    AND p.objid IN (
        '587724197201248493',
        '587724197201313886',
        '587724197201379409',
        '587724197201379418',
        '587724197201379467',
        '587724197201510536',
        '587724197201510561',
        '587724197201510613',
        '587724197202296947',
        '587724197202362516'
    )
    """
    df = mymodule.query_tap(TAP_URL, adql).set_index("dr7objid")
    assert len(df) == 10
    filtered = mymodule.filter_data(df, confidence=0.9)
    assert len(filtered) == 1
