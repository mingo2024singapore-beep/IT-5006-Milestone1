from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

# ========= 1) 加载与合并 =========
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

# 列名规范化
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(" ", "_").str.replace("-", "_"))

# ========= 2) 无效值统一为缺失 =========
invalid_values = ["?", "Unknown", "Invalid", "unknown", "UNK"]
for col in df.columns:
    df[col] = df[col].replace(invalid_values, pd.NA)

# 去掉 gender 的无效类别，只保留 Male/Female
df = df[df["gender"].isin(["Male", "Female"])]

# ========= 3) 删除高缺失列 =========
drop_cols = ["weight", "max_glu_serum", "a1cresult"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ========= 4) 数值列先统一为 numeric（保留 NA）=========
numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    "admission_type_id", "discharge_disposition_id", "admission_source_id"
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ========= 5) 缺失值填充 =========
# 分类型：用众数；数值型：用中位数
for c in df.columns:
    if df[c].dtype == "object":          # 分类/字符串
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True)[0])
    else:                                 # 数值
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

# 数值列转回 Int64（可空整型），保证类型统一
for c in numeric_cols:
    if c in df.columns:
        df[c] = df[c].astype("Int64")

# ========= 6) 构造二分类目标（后续分析/建模更方便）=========
df["readmitted_binary"] = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1).astype("Int64")

# ========= 7) 可选：检查剩余缺失 & 导出 =========
print("\n缺失率前10：")
print(df.isna().mean().sort_values(ascending=False).head(10))

# 导出（按需改路径）
out = "/Users/mingog/Desktop/cleaned_diabetic_final.csv"
df.to_csv(out, index=False, encoding="utf-8")
print(f"\n✅ 干净数据已保存：{out}")
print("形状：", df.shape)
print("dtypes 概览：\n", df.dtypes.value_counts())
