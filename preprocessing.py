from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

repo = fetch_ucirepo(id=296)
X = repo.data.features
y = repo.data.targets

if isinstance(y, pd.Series):
    y = y.rename("readmitted").to_frame()
elif isinstance(y, pd.DataFrame):
    if "readmitted" not in y.columns:
        y = (y.iloc[:, [0]].rename(columns={y.columns[0]: "readmitted"})
             if y.shape[1] > 1 else y.rename(columns={y.columns[0]: "readmitted"}))
else:
    y = pd.Series(np.array(y).ravel(), name="readmitted").to_frame()

df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

df.columns = (df.columns.str.strip().str.lower()
              .str.replace(" ", "_").str.replace("-", "_"))


invalid_values = ["?", "Unknown", "Invalid", "unknown", "UNK"]
for col in df.columns:
    df[col] = df[col].replace(invalid_values, pd.NA)

df = df[df["gender"].isin(["Male", "Female"])]

drop_cols = ["weight", "max_glu_serum", "a1cresult"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    "admission_type_id", "discharge_disposition_id", "admission_source_id"
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for c in df.columns:
    if df[c].dtype == "object":         
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True)[0])
    else:                                 
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

for c in numeric_cols:
    if c in df.columns:
        df[c] = df[c].astype("Int64")


df["readmitted_binary"] = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1).astype("Int64")


print("\n缺失率前10：")
print(df.isna().mean().sort_values(ascending=False).head(10))


out = "/Users/mingog/Desktop/cleaned_diabetic_final.csv"
df.to_csv(out, index=False, encoding="utf-8")
print("形状：", df.shape)
print("dtypes 概览：\n", df.dtypes.value_counts())
