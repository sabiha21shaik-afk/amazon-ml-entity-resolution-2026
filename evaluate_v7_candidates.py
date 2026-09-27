import pandas as pd
import re
from collections import defaultdict


# ============================================================
# 1. NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", "", text)

    return text


def create_name_signatures(name):

    name = normalize_text(name)

    if not name:
        return []

    signatures = set()

    signatures.add("full:" + name)

    if len(name) >= 6:
        signatures.add("prefix6:" + name[:6])

    if len(name) >= 8:

        signatures.add("prefix8:" + name[:8])

        signatures.add(
            "edge:" +
            name[:4] +
            name[-4:]
        )

        signatures.add(
            "edge2:" +
            name[:5] +
            name[-3:]
        )

    return signatures


def create_address_signatures(address):

    if pd.isna(address):
        return []

    text = str(address).lower()

    tokens = re.findall(
        r"[a-z0-9]+",
        text
    )

    tokens = [
        token
        for token in tokens
        if len(token) >= 3
    ]

    signatures = set()

    for token in tokens:

        signatures.add(
            "token:" + token
        )

    for i in range(len(tokens) - 1):

        pair = (
            tokens[i] +
            "_" +
            tokens[i + 1]
        )

        signatures.add(
            "pair:" + pair
        )

    return signatures


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading datasets...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Datasets loaded.")


# ============================================================
# 3. BUILD INDEX
# ============================================================

def build_name_index(df):

    index = defaultdict(set)

    count = 0

    for row in df.itertuples(index=False):

        country = str(row.country)

        for signature in create_name_signatures(
            row.business_name
        ):

            key = country + "|" + signature

            index[key].add(
                row.entity_id
            )

        count += 1

        if count % 500000 == 0:
            print(
                "Name index:",
                count
            )

    return index


def build_address_index(df):

    index = defaultdict(set)

    count = 0

    for row in df.itertuples(index=False):

        country = str(row.country)

        for signature in create_address_signatures(
            row.business_address
        ):

            key = country + "|" + signature

            index[key].add(
                row.entity_id
            )

        count += 1

        if count % 500000 == 0:
            print(
                "Address index:",
                count
            )

    return index


print("\nBuilding S2 indexes...")

name_index_s2 = build_name_index(source2)
address_index_s2 = build_address_index(source2)

print("\nBuilding S3 indexes...")

name_index_s3 = build_name_index(source3)
address_index_s3 = build_address_index(source3)

print("\nIndexes ready.")


# ============================================================
# 4. CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    name,
    address,
    country,
    name_index,
    address_index
):

    country = str(country)

    candidates = set()


    # ----------------------------
    # NAME BLOCKS
    # ----------------------------

    name_blocks = []

    for signature in create_name_signatures(name):

        key = country + "|" + signature

        if key in name_index:

            size = len(
                name_index[key]
            )

            name_blocks.append(
                (size, key)
            )

    name_blocks.sort(
        key=lambda x: x[0]
    )

    for size, key in name_blocks[:2]:

        candidates.update(
            name_index[key]
        )


    # ----------------------------
    # ADDRESS BLOCKS
    # ----------------------------

    address_blocks = []

    for signature in create_address_signatures(address):

        key = country + "|" + signature

        if key in address_index:

            size = len(
                address_index[key]
            )

            address_blocks.append(
                (size, key)
            )

    address_blocks.sort(
        key=lambda x: x[0]
    )

    for size, key in address_blocks[:3]:

        candidates.update(
            address_index[key]
        )

    return candidates


# ============================================================
# 5. MEASURE CANDIDATE COUNTS
# ============================================================

print("\nMeasuring candidate counts...")

s2_total = 0
s3_total = 0

s2_max = 0
s3_max = 0

s2_count = 0
s3_count = 0

total_rows = len(source1)

for row_number, row in enumerate(
    source1.itertuples(index=False),
    start=1
):

    if row_number % 100000 == 0:

        print(
            f"Progress: {row_number:,} / "
            f"{total_rows:,}"
        )


    # ========================================================
    # S2
    # ========================================================

    candidates_s2 = generate_candidates(
        row.business_name,
        row.business_address,
        row.country,
        name_index_s2,
        address_index_s2
    )

    count_s2 = len(candidates_s2)

    s2_total += count_s2
    s2_count += 1

    if count_s2 > s2_max:
        s2_max = count_s2


    # ========================================================
    # S3
    # ========================================================

    candidates_s3 = generate_candidates(
        row.business_name,
        row.business_address,
        row.country,
        name_index_s3,
        address_index_s3
    )

    count_s3 = len(candidates_s3)

    s3_total += count_s3
    s3_count += 1

    if count_s3 > s3_max:
        s3_max = count_s3


# ============================================================
# 6. RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("V7 CANDIDATE VOLUME")
print("=" * 60)

print("\nS2 total candidates:", s2_total)

print(
    "S2 average candidates per S1:",
    round(
        s2_total / s2_count,
        2
    )
)

print(
    "S2 maximum candidates for one S1:",
    s2_max
)


print("\nS3 total candidates:", s3_total)

print(
    "S3 average candidates per S1:",
    round(
        s3_total / s3_count,
        2
    )
)

print(
    "S3 maximum candidates for one S1:",
    s3_max
)


combined_total = (
    s2_total +
    s3_total
)

print(
    "\nCombined average candidates per S1:",
    round(
        combined_total /
        s2_count,
        2
    )
)

print("\nV7 candidate-volume analysis completed.")