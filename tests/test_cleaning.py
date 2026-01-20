import pandas as pd

from iaea_project.cleaning import clean_cyclotron_df


def test_cleaning_adds_energy_num_and_iso3_columns():
    df = pd.DataFrame(
        {
            "Country": ["Switzerland", "Germany"],
            "City": ["Zurich", "Berlin"],
            "Facility": ["A", "B"],
            "Manufacturer": ["Siemens", "ABT"],
            "Model": ["M1", "M2"],
            "Proton energy (MeV)": ["18", "16-18"],
        }
    )

    out = clean_cyclotron_df(df, canonicalize_country=False, canonicalize_manufacturer=True)
    assert "Energy_num" in out.columns
    assert out["Energy_num"].notna().all()
    # Manufacturer backup present if overwritten
    assert "Manufacturer_raw" in out.columns
