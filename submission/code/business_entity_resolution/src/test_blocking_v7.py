import pandas as pd
import re
from collections import defaultdict, Counter


# ============================================================
# 1. NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def get_tokens(text):
    if pd.isna(text):
        return set()

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return {
        x for x in text.split()
        if len(x) >= 4
    }


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

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Datasets loaded.")


# ============================================================
# 3. LOOKUPS
# ============================================================

def create_lookup(df):

    lookup = {}

    for row in df.itertuples(index=False):

        lookup[row.entity_id] = {
            "name": row.business_name,
            "address": row.business_address,
            "country": row.country
        }

    return lookup


print("Creating lookups...")

source1_lookup = create_lookup(source1)
source2_lookup = create_lookup(source2)
source3_lookup = create_lookup(source3)

print("Lookups created.")


# ============================================================
# 4. NAME SIGNATURES
# ============================================================

def create_name_signatures(name):

    name = normalize_text(name)

    if not name:
        return []

    signatures = set()

    signatures.add("full:" + name)

    if len(name) >= 6:
        signatures.add(
            "prefix6:" + name[:6]
        )

    if len(name) >= 8:
        signatures.add(
            "prefix8:" + name[:8]
        )

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


# ============================================================
# 5. ADDRESS SIGNATURES
# ============================================================

def create_address_signatures(address):

    if pd.isna(address):
        return []

    text = str(address).lower()

    # Extract useful alphanumeric tokens
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

    # Individual useful tokens
    for token in tokens:
        signatures.add(
            "token:" + token
        )

    # Adjacent token pairs
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
# 6. BUILD NAME INDEX
# ============================================================

def build_name_index(df):

    index = defaultdict(set)

    count = 0

    for row in df.itertuples(index=False):

        country = str(row.country)

        for signature in create_name_signatures(
            row.business_name
        ):

            key = (
                country +
                "|" +
                signature
            )

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


print("\nBuilding S2 name index...")
name_index_s2 = build_name_index(source2)

print("\nBuilding S3 name index...")
name_index_s3 = build_name_index(source3)


# ============================================================
# 7. BUILD ADDRESS INDEX
# ============================================================

def build_address_index(df):

    index = defaultdict(set)

    count = 0

    for row in df.itertuples(index=False):

        country = str(row.country)

        signatures = create_address_signatures(
            row.business_address
        )

        for signature in signatures:

            key = (
                country +
                "|" +
                signature
            )

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


print("\nBuilding S2 address index...")
address_index_s2 = build_address_index(source2)

print("\nBuilding S3 address index...")
address_index_s3 = build_address_index(source3)

print("Address indexes completed.")


# ============================================================
# 8. TOKEN FREQUENCY
# ============================================================

def calculate_token_frequency(df):

    frequency = Counter()

    for row in df.itertuples(index=False):

        tokens = get_tokens(
            row.business_name
        )

        for token in tokens:
            frequency[token] += 1

    return frequency


print("\nCalculating name token frequencies...")

frequency_s2 = calculate_token_frequency(source2)
frequency_s3 = calculate_token_frequency(source3)


# ============================================================
# 9. CANDIDATE GENERATION
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

    # --------------------------------------------------------
    # NAME BLOCKS
    # --------------------------------------------------------

    name_blocks = []

    for signature in create_name_signatures(name):

        key = (
            country +
            "|" +
            signature
        )

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

    # Take 2 smallest name blocks
    for size, key in name_blocks[:2]:

        candidates.update(
            name_index[key]
        )


    # --------------------------------------------------------
    # ADDRESS BLOCKS
    # --------------------------------------------------------

    address_blocks = []

    for signature in create_address_signatures(
        address
    ):

        key = (
            country +
            "|" +
            signature
        )

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

    # Take 3 smallest address blocks
    for size, key in address_blocks[:3]:

        candidates.update(
            address_index[key]
        )

    return candidates


# ============================================================
# 10. EVALUATION
# ============================================================

print("\nStarting V7 evaluation...")

true_s2 = 0
true_s3 = 0

captured_s2 = 0
captured_s3 = 0

total_s1 = len(ground_truth)

for row_number, row in enumerate(
    ground_truth.itertuples(index=False),
    start=1
):

    if row_number % 100000 == 0:

        print(
            f"Progress: {row_number:,} / "
            f"{total_s1:,}"
        )

    s1_id = row.source1_entity_id

    if s1_id not in source1_lookup:
        continue

    s1 = source1_lookup[s1_id]

    matched_ids = row.matched_entity_ids

    if pd.isna(matched_ids):
        continue

    true_ids = str(
        matched_ids
    ).split(",")


    # ========================================================
    # S2
    # ========================================================

    true_s2_ids = [
        x for x in true_ids
        if x.startswith("S2-")
    ]

    true_s2 += len(true_s2_ids)

    if true_s2_ids:

        candidates = generate_candidates(
            s1["name"],
            s1["address"],
            s1["country"],
            name_index_s2,
            address_index_s2
        )

        for true_id in true_s2_ids:

            if true_id in candidates:
                captured_s2 += 1


    # ========================================================
    # S3
    # ========================================================

    true_s3_ids = [
        x for x in true_ids
        if x.startswith("S3-")
    ]

    true_s3 += len(true_s3_ids)

    if true_s3_ids:

        candidates = generate_candidates(
            s1["name"],
            s1["address"],
            s1["country"],
            name_index_s3,
            address_index_s3
        )

        for true_id in true_s3_ids:

            if true_id in candidates:
                captured_s3 += 1


# ============================================================
# 11. RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("V7 RESULTS")
print("=" * 60)

print("\nS2 true matches:", true_s2)
print("S2 captured:", captured_s2)

if true_s2 > 0:
    print(
        "S2 recall:",
        round(
            captured_s2 / true_s2 * 100,
            2
        ),
        "%"
    )


print("\nS3 true matches:", true_s3)
print("S3 captured:", captured_s3)

if true_s3 > 0:
    print(
        "S3 recall:",
        round(
            captured_s3 / true_s3 * 100,
            2
        ),
        "%"
    )


total_true = true_s2 + true_s3
total_captured = captured_s2 + captured_s3

print("\nOverall true matches:", total_true)
print("Overall captured:", total_captured)

if total_true > 0:

    print(
        "Overall recall:",
        round(
            total_captured /
            total_true *
            100,
            2
        ),
        "%"
    )

print("\nV7 completed.")