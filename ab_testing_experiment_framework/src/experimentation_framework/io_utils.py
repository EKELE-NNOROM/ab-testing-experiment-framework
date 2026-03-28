from pathlib import Path
import pandas as pd

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
