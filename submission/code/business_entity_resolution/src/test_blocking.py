import pandas as pd
import re


SOURCE1_PATH = "dataset/train/train_source1.tsv"
SOURCE2_PATH = "dataset/train/train_source2.tsv"
SOURCE3_PATH = "dataset/train/train_source3.tsv"


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", "", text)

    return text


print("=" * 60)
print("TESTING BLOCKING STRATEGY")
print("=" * 60)

print("\nLoading Source 1...")
source1 = pd.read_csv(SOURCE1_PATH, sep="\t")

print("Loading Source 2...")
source2 = pd.read_csv(SOURCE2_PATH, sep="\t")

print("Loading Source 3...")
source3 = pd.read_csv(SOURCE3_PATH, sep="\t")


# Normalize names
print("\nNormalizing business names...")

source1["name_key"] = source1["business_name"].apply(normalize_text)
source2["name_key"] = source2["business_name"].apply(normalize_text)
source3["name_key"] = source3["business_name"].apply(normalize_text)


# Create blocking keys
source1["block_key"] = (
    source1["country"].fillna("").astype(str)
    + "|"
    + source1["name_key"]
)

source2["block_key"] = (
    source2["country"].fillna("").astype(str)
    + "|"
    + source2["name_key"]
)

source3["block_key"] = (
    source3["country"].fillna("").astype(str)
    + "|"
    + source3["name_key"]
)


print("\nBuilding lookup tables...")


# Number of Source 2 records for each block
source2_counts = source2["block_key"].value_counts()

# Number of Source 3 records for each block
source3_counts = source3["block_key"].value_counts()


# Attach candidate counts to Source 1
source1["source2_candidates"] = (
    source1["block_key"].map(source2_counts).fillna(0)
)

source1["source3_candidates"] = (
    source1["block_key"].map(source3_counts).fillna(0)
)


# Total candidates
source1["total_candidates"] = (
    source1["source2_candidates"]
    + source1["source3_candidates"]
)


print("\n" + "=" * 60)
print("BLOCKING RESULTS")
print("=" * 60)


print("\nSource 1 records:", len(source1))

print(
    "\nSource 1 records with at least one Source 2 candidate:",
    (source1["source2_candidates"] > 0).sum()
)

print(
    "Source 1 records with at least one Source 3 candidate:",
    (source1["source3_candidates"] > 0).sum()
)

print(
    "Source 1 records with at least one candidate overall:",
    (source1["total_candidates"] > 0).sum()
)


print("\nAverage Source 2 candidates per Source 1:")
print(source1["source2_candidates"].mean())

print("\nAverage Source 3 candidates per Source 1:")
print(source1["source3_candidates"].mean())

print("\nAverage total candidates per Source 1:")
print(source1["total_candidates"].mean())


print("\nMaximum Source 2 candidates for one Source 1:")
print(source1["source2_candidates"].max())

print("\nMaximum Source 3 candidates for one Source 1:")
print(source1["source3_candidates"].max())


print("\n" + "=" * 60)
print("CANDIDATE COUNT DISTRIBUTION")
print("=" * 60)

print("\nSource 2:")
print(source1["source2_candidates"].describe())

print("\nSource 3:")
print(source1["source3_candidates"].describe())


print("\n" + "=" * 60)
print("EXAMPLE BLOCKS")
print("=" * 60)

examples = source1[
    source1["total_candidates"] > 0
][
    [
        "entity_id",
        "business_name",
        "country",
        "source2_candidates",
        "source3_candidates",
        "total_candidates"
    ]
].head(20)

print(examples.to_string(index=False))


print("\n" + "=" * 60)
print("BLOCKING TEST COMPLETE")
print("=" * 60)