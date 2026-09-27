import pandas as pd
import re


SOURCE1_PATH = "dataset/train/train_source1.tsv"
SOURCE2_PATH = "dataset/train/train_source2.tsv"
SOURCE3_PATH = "dataset/train/train_source3.tsv"
GROUND_TRUTH_PATH = "dataset/train/train_ground_truth.tsv"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9]", "", text)

    return text


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 60)
print("LOADING SOURCE 1")
print("=" * 60)

source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

source1["name_key"] = source1["business_name"].apply(
    normalize_text
)

source1["block_key"] = (
    source1["country"].fillna("").astype(str)
    + "|"
    + source1["name_key"]
)


# ============================================================
# LOAD SOURCE 2
# ============================================================

print("\nLoading Source 2...")

source2 = pd.read_csv(
    SOURCE2_PATH,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

source2["name_key"] = source2["business_name"].apply(
    normalize_text
)

source2["block_key"] = (
    source2["country"].fillna("").astype(str)
    + "|"
    + source2["name_key"]
)


# Create entity_id -> block_key lookup
source2_lookup = source2[
    ["entity_id", "block_key"]
].drop_duplicates(
    "entity_id"
)

source2_lookup = source2_lookup.set_index(
    "entity_id"
)["block_key"]


del source2


# ============================================================
# LOAD SOURCE 3
# ============================================================

print("Loading Source 3...")

source3 = pd.read_csv(
    SOURCE3_PATH,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

source3["name_key"] = source3["business_name"].apply(
    normalize_text
)

source3["block_key"] = (
    source3["country"].fillna("").astype(str)
    + "|"
    + source3["name_key"]
)

source3_lookup = source3[
    ["entity_id", "block_key"]
].drop_duplicates(
    "entity_id"
)

source3_lookup = source3_lookup.set_index(
    "entity_id"
)["block_key"]


del source3


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t"
)


# ============================================================
# CREATE SOURCE 1 BLOCK LOOKUP
# ============================================================

source1_blocks = source1[
    ["entity_id", "block_key"]
].set_index("entity_id")["block_key"]


# ============================================================
# EVALUATE TRUE MATCHES
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING BLOCKING RECALL")
print("=" * 60)


total_true_matches = 0
captured_true_matches = 0

source2_total = 0
source2_captured = 0

source3_total = 0
source3_captured = 0


for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    matched_ids = row["matched_entity_ids"]

    if pd.isna(matched_ids):
        continue

    source1_block = source1_blocks.get(source1_id)

    if pd.isna(source1_block):
        continue

    matched_ids = str(matched_ids).split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):

            source2_total += 1

            matched_block = source2_lookup.get(
                matched_id
            )

        elif matched_id.startswith("S3-"):

            source3_total += 1

            matched_block = source3_lookup.get(
                matched_id
            )

        else:
            continue

        if pd.isna(matched_block):
            continue

        total_true_matches += 1

        if matched_block == source1_block:

            captured_true_matches += 1

            if matched_id.startswith("S2-"):
                source2_captured += 1

            else:
                source3_captured += 1


# ============================================================
# RESULTS
# ============================================================

print("\nTotal true matched pairs:")
print(total_true_matches)

print("\nTrue Source 2 matches:")
print(source2_total)

print("Captured by current blocking:")
print(source2_captured)

if source2_total > 0:
    print(
        "Source 2 blocking recall:",
        round(
            source2_captured / source2_total * 100,
            2
        ),
        "%"
    )


print("\nTrue Source 3 matches:")
print(source3_total)

print("Captured by current blocking:")
print(source3_captured)

if source3_total > 0:
    print(
        "Source 3 blocking recall:",
        round(
            source3_captured / source3_total * 100,
            2
        ),
        "%"
    )


print("\nOverall captured:")
print(captured_true_matches)

if total_true_matches > 0:

    print(
        "\nOverall blocking recall:",
        round(
            captured_true_matches / total_true_matches * 100,
            2
        ),
        "%"
    )


print("\n" + "=" * 60)
print("BLOCKING RECALL TEST COMPLETE")
print("=" * 60)