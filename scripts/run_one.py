from iaea_project.utils import ensure_dirs, RAW_DIR
from iaea_project.cleaning import clean_cyclotron_df
import pandas as pd

def main():
    ensure_dirs()
    raw_csv = RAW_DIR / "iaea_cyclotrons_raw.csv"
    df = pd.read_csv(raw_csv)
    df2 = clean_cyclotron_df(df)
    print(df2.head())

if __name__ == "__main__":
    main()
