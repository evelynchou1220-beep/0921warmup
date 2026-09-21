#!/usr/bin/env python3
"""把證交所「每日收盤行情(ETF)」原始 CSV 整理成分析用資料。

用法:
    python scripts/process.py data/raw/MI_INDEX_0099P_20260918.csv
    python scripts/process.py data/raw/MI_INDEX_0099P_20260918.csv --min-amount 2

產出:
    data/processed/YYYY-MM-DD.csv   當天整理後的資料（含計算欄位）
    history/etf_daily.csv           所有日期累積的長表（同一天重跑會覆蓋，不會重複）
    reports/YYYY-MM-DD.md           當天文字摘要
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_NUM = ["成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差"]
OUT_COLS = ["日期", "證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
            "開盤價", "最高價", "最低價", "收盤價", "漲跌方向", "漲跌價差",
            "前收盤價", "漲跌幅", "日內漲跌幅", "日內振幅", "收盤位置", "成交金額_億元"]


def parse_date(path: Path) -> str:
    """從檔案第一行的民國日期（例如 115年09月18日）轉成西元 YYYY-MM-DD。"""
    first = path.open(encoding="cp950").readline()
    m = re.search(r"(\d+)年(\d+)月(\d+)日", first)
    if not m:
        raise ValueError(f"找不到日期: {first!r}")
    y, mo, d = map(int, m.groups())
    return f"{y + 1911:04d}-{mo:02d}-{d:02d}"


def to_num(s: pd.Series) -> pd.Series:
    s = s.str.replace(",", "", regex=False).str.strip().replace({"--": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")


def read_raw(path: Path) -> pd.DataFrame:
    # 證交所檔案為 cp950（Big5 擴充）編碼；前兩行是標題，「備註:」之後是頁尾說明
    df = pd.read_csv(path, skiprows=2, encoding="cp950", dtype=str)
    foot = df.index[df["證券代號"].str.startswith("備註", na=False)]
    if len(foot):
        df = df.loc[: foot[0] - 1]
    df["證券代號"] = df["證券代號"].str.strip()
    df["證券名稱"] = df["證券名稱"].str.strip()
    return df


def transform(df: pd.DataFrame, date: str) -> pd.DataFrame:
    for c in RAW_NUM:
        df[c] = to_num(df[c])
    sign = df["漲跌(+/-)"].fillna("").str.strip()
    df["漲跌方向"] = sign.replace("", np.nan)

    close, diff = df["收盤價"], df["漲跌價差"]
    # 前收盤價：漲則 收盤−價差；跌則 收盤+價差；平盤=收盤；X(不比價)無前收
    df["前收盤價"] = np.where(sign == "+", close - diff,
                     np.where(sign == "-", close + diff,
                     np.where(sign == "X", np.nan, close)))
    df["漲跌幅"] = (close - df["前收盤價"]) / df["前收盤價"]
    df["日內漲跌幅"] = (close - df["開盤價"]) / df["開盤價"]
    df["日內振幅"] = (df["最高價"] - df["最低價"]) / df["開盤價"]
    rng = (df["最高價"] - df["最低價"]).replace(0, np.nan)
    df["收盤位置"] = (close - df["最低價"]) / rng
    df["成交金額_億元"] = df["成交金額"] / 1e8
    df.insert(0, "日期", date)

    for c in ["漲跌幅", "日內漲跌幅", "日內振幅", "收盤位置", "成交金額_億元", "前收盤價"]:
        df[c] = df[c].round(6)
    return df[OUT_COLS]


def md_table(d: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "|".join("---" for _ in d.columns) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def build_report(d: pd.DataFrame, date: str, min_amount: float) -> str:
    chg = d["漲跌幅"]
    up, down, flat = int((chg > 0).sum()), int((chg < 0).sum()), int((chg == 0).sum())
    fmt = lambda x: "-" if pd.isna(x) else f"{x:+.2%}"

    top_amt = d.nlargest(10, "成交金額_億元")[["證券代號", "證券名稱", "成交金額_億元", "漲跌幅"]].copy()
    top_amt["成交金額_億元"] = top_amt["成交金額_億元"].round(1)
    top_amt["漲跌幅"] = top_amt["漲跌幅"].map(fmt)

    liquid = d[(d["成交金額_億元"] >= min_amount) & d["漲跌幅"].notna()]
    cols = ["證券代號", "證券名稱", "漲跌幅", "成交金額_億元"]
    best = liquid.nlargest(10, "漲跌幅")[cols].copy()
    worst = liquid.nsmallest(10, "漲跌幅")[cols].copy()
    for t in (best, worst):
        t["漲跌幅"] = t["漲跌幅"].map(fmt)
        t["成交金額_億元"] = t["成交金額_億元"].round(1)

    return f"""# ETF 每日收盤摘要 {date}

## 市場概況
- 總檔數：{len(d)}，有成交：{int((d['成交股數'] > 0).sum())}
- 上漲 {up} 檔／下跌 {down} 檔／平盤 {flat} 檔／不比價(X) {int((d['漲跌方向'] == 'X').sum())} 檔
- 總成交金額：{d['成交金額_億元'].sum():,.1f} 億元
- 漲跌幅中位數：{fmt(chg.median())}，平均：{fmt(chg.mean())}

## 成交金額前 10 名
{md_table(top_amt)}

## 漲跌幅最高 10 名（成交金額 ≥ {min_amount:g} 億元）
{md_table(best)}

## 漲跌幅最低 10 名（成交金額 ≥ {min_amount:g} 億元）
{md_table(worst)}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_csv", type=Path)
    ap.add_argument("--min-amount", type=float, default=1.0, help="排行的最低成交金額門檻（億元），預設 1")
    a = ap.parse_args()

    date = parse_date(a.raw_csv)
    d = transform(read_raw(a.raw_csv), date)

    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "history").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    d.to_csv(ROOT / f"data/processed/{date}.csv", index=False, encoding="utf-8-sig")

    hist_path = ROOT / "history/etf_daily.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path, dtype={"證券代號": str}, encoding="utf-8-sig")
        hist = hist[hist["日期"] != date]
        hist = pd.concat([hist, d], ignore_index=True)
    else:
        hist = d
    hist.sort_values(["日期", "證券代號"]).to_csv(hist_path, index=False, encoding="utf-8-sig")

    (ROOT / f"reports/{date}.md").write_text(build_report(d, date, a.min_amount), encoding="utf-8")
    print(f"{date}: {len(d)} 檔 → data/processed/{date}.csv、history/etf_daily.csv（共 {hist['日期'].nunique()} 天）、reports/{date}.md")


if __name__ == "__main__":
    main()
