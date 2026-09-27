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

    # Keep only letters and numbers
    text = re.sub(r"[^a-z0-9]", "", text)

    return text


# ============================================================
# CREATE BLOCK KEY
# ============================================================

def create_block_key(df):

    name_key = df["business_name"].apply(normalize_text)
    address_key = df["business_address"].apply(normalize_text)

    country_key = df["country"].fillna("").astype(str)

    # Address-based blocking key
    address_block = (
        country_key
        + "|"
        + address_key
    )

    # Name-based blocking key
    name_block = (
        country_key
        + "|"
        + name_key
    )

    return name_block, address_block


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
        "business_address",
        "country"
    ]
)

source1["name_block"], source1["address_block"] = (
    create_block_key(source1)
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
        "business_address",
        "country"
    ]
)

source2["name_block"], source2["address_block"] = (
    create_block_key(source2)
)


# Create lookup tables

source2_name_lookup = (
    source2[
        ["entity_id", "name_block"]
    ]
    .drop_duplicates("entity_id")
    .set_index("entity_id")["name_block"]
)

source2_address_lookup = (
    source2[
        ["entity_id", "address_block"]
    ]
    .drop_duplicates("entity_id")
    .set_index("entity_id")["address_block"]
)


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
        "business_address",
        "country"
    ]
)

source3["name_block"], source3["address_block"] = (
    create_block_key(source3)
)


source3_name_lookup = (
    source3[
        ["entity_id", "name_block"]
    ]
    .drop_duplicates("entity_id")
    .set_index("entity_id")["name_block"]
)

source3_address_lookup = (
    source3[
        ["entity_id", "address_block"]
    ]
    .drop_duplicates("entity_id")
    .set_index("entity_id")["address_block"]
)


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
# SOURCE 1 LOOKUPS
# ============================================================

source1_name_lookup = (
    source1
    .set_index("entity_id")["name_block"]
)

source1_address_lookup = (
    source1
    .set_index("entity_id")["address_block"]
)


# ============================================================
# BLOCKING RECALL TEST
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING NAME + ADDRESS BLOCKING")
print("=" * 60)


total_matches = 0

name_captured = 0
address_captured = 0
union_captured = 0

source2_total = 0
source2_name = 0
source2_address = 0
source2_union = 0

source3_total = 0
source3_name = 0
source3_address = 0
source3_union = 0


for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]
    matched_ids = row["matched_entity_ids"]

    if pd.isna(matched_ids):
        continue

    s1_name_block = source1_name_lookup.get(source1_id)
    s1_address_block = source1_address_lookup.get(source1_id)

    if pd.isna(s1_name_block):
        continue

    matched_ids = str(matched_ids).split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        # ---------------------------------------------
        # SOURCE 2
        # ---------------------------------------------

        if matched_id.startswith("S2-"):

            source2_total += 1

            candidate_name_block = (
                source2_name_lookup.get(matched_id)
            )

            candidate_address_block = (
                source2_address_lookup.get(matched_id)
            )

        # ---------------------------------------------
        # SOURCE 3
        # ---------------------------------------------

        elif matched_id.startswith("S3-"):

            source3_total += 1

            candidate_name_block = (
                source3_name_lookup.get(matched_id)
            )

            candidate_address_block = (
                source3_address_lookup.get(matched_id)
            )

        else:
            continue


        # Empty address blocks should NOT count
        address_match = (
            s1_address_block != ""
            and candidate_address_block != ""
            and s1_address_block == candidate_address_block
        )

        name_match = (
            s1_name_block != ""
            and candidate_name_block != ""
            and s1_name_block == candidate_name_block
        )


        if name_match:
            name_captured += 1

        if address_match:
            address_captured += 1

        if name_match or address_match:
            union_captured += 1


        # Per-source statistics

        if matched_id.startswith("S2-"):

            if name_match:
                source2_name += 1

            if address_match:
                source2_address += 1

            if name_match or address_match:
                source2_union += 1

        else:

            if name_match:
                source3_name += 1

            if address_match:
                source3_address += 1

            if name_match or address_match:
                source3_union += 1


        total_matches += 1


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)


print("\nTotal true matches:", total_matches)


print("\nSOURCE 2")
print("-" * 40)

print("True matches:", source2_total)

print("Name block captured:", source2_name)

print("Address block captured:", source2_address)

print("Name + Address union captured:", source2_union)

print(
    "Name recall:",
    round(source2_name / source2_total * 100, 2),
    "%"
)

print(
    "Address recall:",
    round(source2_address / source2_total * 100, 2),
    "%"
)

print(
    "Union recall:",
    round(source2_union / source2_total * 100, 2),
    "%"
)


print("\nSOURCE 3")
print("-" * 40)

print("True matches:", source3_total)

print("Name block captured:", source3_name)

print("Address block captured:", source3_address)

print("Name + Address union captured:", source3_union)

print(
    "Name recall:",
    round(source3_name / source3_total * 100, 2),
    "%"
)

print(
    "Address recall:",
    round(source3_address / source3_total * 100, 2),
    "%"
)

print(
    "Union recall:",
    round(source3_union / source3_total * 100, 2),
    "%"
)


print("\nOVERALL")
print("-" * 40)

print("Name block captured:", name_captured)

print("Address block captured:", address_captured)

print("Name + Address union captured:", union_captured)

print(
    "Name recall:",
    round(name_captured / total_matches * 100, 2),
    "%"
)

print(
    "Address recall:",
    round(address_captured / total_matches * 100, 2),
    "%"
)

print(
    "Union recall:",
    round(union_captured / total_matches * 100, 2),
    "%"
)


print("\n" + "=" * 60)
print("ADDRESS BLOCKING TEST COMPLETE")
print("=" * 60)