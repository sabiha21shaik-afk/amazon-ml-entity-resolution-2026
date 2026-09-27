import pandas as pd
import re


# ============================================================
# SETTINGS
# ============================================================

SOURCE1_PATH = "dataset/train/train_source1.tsv"
SOURCE2_PATH = "dataset/train/train_source2.tsv"
SOURCE3_PATH = "dataset/train/train_source3.tsv"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Converts business text into a simpler form.

    Example:
        "ABC Pvt. Ltd." -> "abcpvtltd"
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove anything that is not a letter or number
    text = re.sub(r"[^a-z0-9]", "", text)

    return text


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

source1 = pd.read_csv(SOURCE1_PATH, sep="\t")
source2 = pd.read_csv(SOURCE2_PATH, sep="\t")
source3 = pd.read_csv(SOURCE3_PATH, sep="\t")


print("\nRows loaded:")
print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))


# ============================================================
# NORMALIZE BUSINESS NAMES
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZING BUSINESS NAMES")
print("=" * 60)

source1["name_normalized"] = source1["business_name"].apply(normalize_text)
source2["name_normalized"] = source2["business_name"].apply(normalize_text)
source3["name_normalized"] = source3["business_name"].apply(normalize_text)


# ============================================================
# EXACT NAME OVERLAP
# ============================================================

print("\n" + "=" * 60)
print("EXACT NORMALIZED NAME OVERLAP")
print("=" * 60)


# Remove empty names
s1_names = set(source1.loc[
    source1["name_normalized"] != "",
    "name_normalized"
])

s2_names = set(source2.loc[
    source2["name_normalized"] != "",
    "name_normalized"
])

s3_names = set(source3.loc[
    source3["name_normalized"] != "",
    "name_normalized"
])


print("\nUnique Source 1 names:", len(s1_names))
print("Unique Source 2 names:", len(s2_names))
print("Unique Source 3 names:", len(s3_names))


common_s1_s2 = s1_names & s2_names
common_s1_s3 = s1_names & s3_names

print("\nNames appearing in both Source 1 and Source 2:",
      len(common_s1_s2))

print("Names appearing in both Source 1 and Source 3:",
      len(common_s1_s3))


# ============================================================
# COUNTRY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("COUNTRY DISTRIBUTION")
print("=" * 60)

print("\nSource 1:")
print(source1["country"].value_counts().head(20))

print("\nSource 2:")
print(source2["country"].value_counts().head(20))

print("\nSource 3:")
print(source3["country"].value_counts().head(20))


# ============================================================
# SAMPLE NORMALIZED NAMES
# ============================================================

print("\n" + "=" * 60)
print("NAME NORMALIZATION EXAMPLES")
print("=" * 60)

examples = source1[
    ["business_name", "name_normalized"]
].head(20)

print(examples.to_string(index=False))


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)