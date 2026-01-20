import pandas as pd
from iaea_project.cleaning import clean_cyclotron_df

def test_cleaning_does_not_drop_all_rows():
    df = pd.DataFrame({"Country": ["Switzerland", "Germany"], "Manufacturer": ["A", "B"]})
    out = clean_cyclotron_df(df)
    assert len(out) > 0
