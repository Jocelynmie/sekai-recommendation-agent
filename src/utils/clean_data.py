# src/utils/clean_data.py
import pandas as pd, os, re, unicodedata, html, json
RAW_DIR, PROC_DIR = "data/raw", "data/processed"; os.makedirs(PROC_DIR, exist_ok=True)

def _norm(t:str)->str:
    t = html.unescape(str(t)); t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t); return t.strip().lower()

def _split(series):               # 列表字段切分
    return series.fillna("").apply(lambda s: [x.strip().lower() for x in re.split(r"[;,]", s) if x.strip()])

def main():
    users    = pd.read_csv(f"{RAW_DIR}/users.csv")
    contents = pd.read_csv(f"{RAW_DIR}/contents.csv")

    users["user_interest_tags"] = _split(users["user_interest_tags"])
    contents["character_list"]  = _split(contents["character_list"])

    for col in ["title", "intro", "initial_record"]:
        contents[col] = contents[col].apply(_norm)

    users.to_parquet(f"{PROC_DIR}/users_clean.parquet", index=False)
    contents.to_parquet(f"{PROC_DIR}/contents_clean.parquet", index=False)
    print("✅ clean parquet files written to data/processed/")
if __name__ == "__main__": main()
