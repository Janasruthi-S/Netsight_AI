import pandas as pd
import os
import numpy as np

# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = "dataset/combined_network_dataset.csv"

print("=" * 70)
print("             NETWORK IDS DATASET ANALYSIS")
print("=" * 70)

# Check whether file exists
if not os.path.exists(DATA_PATH):
    print("\n❌ Dataset not found!")
    print("Expected path:", DATA_PATH)
    print("\nFiles available in dataset folder:")

    if os.path.exists("dataset"):
        for file in os.listdir("dataset"):
            print("  -", file)
    else:
        print("  dataset folder does not exist")

    raise SystemExit

print("\n✅ Dataset found:", DATA_PATH)

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully!")

# ============================================================
# 1. SHAPE
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET SHAPE")
print("=" * 70)

rows, columns = df.shape

print("Number of rows    :", rows)
print("Number of columns :", columns)

# ============================================================
# 2. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("2. COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns):
    print(f"{i+1:3}. {column}")

# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ============================================================
# 4. LABEL COLUMN
# ============================================================

print("\n" + "=" * 70)
print("4. LABEL COLUMN")
print("=" * 70)

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()

if "Label" in df.columns:

    print("✅ Label column found")

    print("\nClass distribution:")
    print(df["Label"].value_counts())

    print("\nNumber of classes:")
    print(df["Label"].nunique())

else:

    print("❌ 'Label' column was NOT found.")

    print("\nAvailable columns:")
    print(list(df.columns))

# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

total_missing = missing.sum()

print("Total missing values:", total_missing)

if total_missing > 0:

    print("\nColumns containing missing values:")

    print(
        missing[missing > 0]
        .sort_values(ascending=False)
    )

else:

    print("✅ No missing values found")

# ============================================================
# 6. DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("6. DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates == 0:
    print("✅ No duplicate rows")
else:
    print("⚠️ Duplicate rows detected")

# ============================================================
# 7. INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("7. INFINITE VALUES")
print("=" * 70)

numeric_df = df.select_dtypes(include=[np.number])

infinite_count = np.isinf(numeric_df).sum().sum()

print("Infinite numeric values:", infinite_count)

if infinite_count == 0:
    print("✅ No infinite values")
else:
    print("⚠️ Infinite values detected")

# ============================================================
# 8. NUMERIC / NON-NUMERIC COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("8. FEATURE TYPES")
print("=" * 70)

numeric_columns = df.select_dtypes(include=[np.number]).columns
non_numeric_columns = df.select_dtypes(
    exclude=[np.number]
).columns

print("\nNumeric columns:", len(numeric_columns))

for column in numeric_columns:
    print("  -", column)

print("\nNon-numeric columns:", len(non_numeric_columns))

for column in non_numeric_columns:
    print("  -", column)

# ============================================================
# 9. MEMORY USAGE
# ============================================================

print("\n" + "=" * 70)
print("9. MEMORY USAGE")
print("=" * 70)

memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

print(f"Dataset memory usage: {memory_mb:.2f} MB")

# ============================================================
# 10. FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("10. FIRST 5 ROWS")
print("=" * 70)

print(df.head())

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("              DATASET CHECK COMPLETE")
print("=" * 70)