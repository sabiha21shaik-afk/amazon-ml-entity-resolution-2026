import pandas as pd
import re
from collections import defaultdict, Counter


# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================

def normalize_name(text):
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
        token
        for token in text.split()
        if len(token) >= 4
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
# 3. CREATE LOOKUPS
# ============================================================

print("\nCreating lookups...")


def create_lookup(df):

    lookup = {}

    for row in df.itertuples(index=False):

        lookup[row.entity_id] = {
            "name": row.business_name,
            "address": row.business_address,
            "country": row.country
        }

    return lookup


source1_lookup = create_lookup(source1)
source2_lookup = create_lookup(source2)
source3_lookup = create_lookup(source3)

print("Lookups created.")


# ============================================================
# 4. TOKEN FREQUENCY
# ============================================================

print("\nCalculating token frequencies...")


def calculate_frequency(df):

    frequency = Counter()

    for row in df.itertuples(index=False):

        tokens = get_tokens(row.business_name)

        for token in tokens:
            frequency[token] += 1

    return frequency


frequency_s2 = calculate_frequency(source2)
print("S2 token frequencies completed.")

frequency_s3 = calculate_frequency(source3)
print("S3 token frequencies completed.")


# ============================================================
# 5. NAME SIGNATURES
# ============================================================

def create_signatures(name):

    name = normalize_name(name)

    if not name:
        return []

    signatures = set()

    signatures.add("full:" + name)
    signatures.add("prefix6:" + name[:6])
    signatures.add("prefix8:" + name[:8])

    if len(name) >= 8:

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
# 6. BUILD NAME INDEX
# ============================================================

def build_name_index(df):

    index = defaultdict(set)

    count = 0

    for row in df.itertuples(index=False):

        country = str(row.country)

        signatures = create_signatures(
            row.business_name
        )

        for signature in signatures:

            key = country + "|" + signature

            index[key].add(
                row.entity_id
            )

        count += 1

        if count % 500000 == 0:
            print(
                "Name index progress:",
                count,
                "rows"
            )

    return index


print("\nBuilding S2 name index...")
name_index_s2 = build_name_index(source2)

print("\nBuilding S3 name index...")
name_index_s3 = build_name_index(source3)

print("Name indexes completed.")


# ============================================================
# 7. BUILD TOKEN INDEX
# ============================================================

def build_token_index(df, frequency):

    index = defaultdict(set)

    count = 0

    for row in df.itertuples(index=False):

        country = str(row.country)

        tokens = get_tokens(
            row.business_name
        )

        for token in tokens:

            if frequency[token] <= 1000:

                key = (
                    country +
                    "|" +
                    token
                )

                index[key].add(
                    row.entity_id
                )

        count += 1

        if count % 500000 == 0:
            print(
                "Token index progress:",
                count,
                "rows"
            )

    return index


print("\nBuilding S2 token index...")
token_index_s2 = build_token_index(
    source2,
    frequency_s2
)

print("\nBuilding S3 token index...")
token_index_s3 = build_token_index(
    source3,
    frequency_s3
)

print("Token indexes completed.")


# ============================================================
# 8. CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    business_name,
    country,
    name_index,
    token_index,
    frequency
):

    country = str(country)

    candidates = set()

    # ----------------------------
    # NAME BLOCKS
    # ----------------------------

    signatures = create_signatures(
        business_name
    )

    name_blocks = []

    for signature in signatures:

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

    for size, key in name_blocks[:2]:

        candidates.update(
            name_index[key]
        )

    # ----------------------------
    # TOKEN BLOCKS
    # ----------------------------

    tokens = get_tokens(
        business_name
    )

    token_blocks = []

    for token in tokens:

        freq = frequency.get(
            token,
            999999999
        )

        if freq <= 1000:

            key = (
                country +
                "|" +
                token
            )

            if key in token_index:

                size = len(
                    token_index[key]
                )

                token_blocks.append(
                    (freq, size, key)
                )

    token_blocks.sort(
        key=lambda x: x[0]
    )

    for freq, size, key in token_blocks[:2]:

        candidates.update(
            token_index[key]
        )

    return candidates


# ============================================================
# 9. ANALYZE MISSED MATCHES
# ============================================================

print("\nStarting missed-match analysis...")

missed_s2 = 0
missed_s3 = 0

shown_s2 = 0
shown_s3 = 0

MAX_EXAMPLES = 20


# ============================================================
# 10. PROCESS GROUND TRUTH
# ============================================================

total_rows = len(ground_truth)

print(
    "Total ground-truth rows:",
    total_rows
)

for row_number, row in enumerate(
    ground_truth.itertuples(index=False),
    start=1
):

    # ------------------------------------
    # PROGRESS MESSAGE
    # ------------------------------------

    if row_number % 100000 == 0:

        print(
            f"Progress: {row_number:,} / "
            f"{total_rows:,} rows"
        )

        print(
            "Missed S2:",
            missed_s2,
            "| Missed S3:",
            missed_s3
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
    # SOURCE 2
    # ========================================================

    true_s2_ids = [
        x
        for x in true_ids
        if x.startswith("S2-")
    ]

    if true_s2_ids:

        candidates_s2 = generate_candidates(
            s1["name"],
            s1["country"],
            name_index_s2,
            token_index_s2,
            frequency_s2
        )

        for true_id in true_s2_ids:

            if true_id not in candidates_s2:

                missed_s2 += 1

                if shown_s2 < MAX_EXAMPLES:

                    if true_id in source2_lookup:

                        target = source2_lookup[
                            true_id
                        ]

                        print("\n")
                        print("=" * 60)
                        print(
                            f"MISSED S2 EXAMPLE "
                            f"#{shown_s2 + 1}"
                        )
                        print("=" * 60)

                        print(
                            "S1 name:",
                            s1["name"]
                        )

                        print(
                            "S2 name:",
                            target["name"]
                        )

                        print(
                            "S1 address:",
                            s1["address"]
                        )

                        print(
                            "S2 address:",
                            target["address"]
                        )

                        print(
                            "Common tokens:",
                            get_tokens(
                                s1["name"]
                            ) &
                            get_tokens(
                                target["name"]
                            )
                        )

                        shown_s2 += 1


    # ========================================================
    # SOURCE 3
    # ========================================================

    true_s3_ids = [
        x
        for x in true_ids
        if x.startswith("S3-")
    ]

    if true_s3_ids:

        candidates_s3 = generate_candidates(
            s1["name"],
            s1["country"],
            name_index_s3,
            token_index_s3,
            frequency_s3
        )

        for true_id in true_s3_ids:

            if true_id not in candidates_s3:

                missed_s3 += 1

                if shown_s3 < MAX_EXAMPLES:

                    if true_id in source3_lookup:

                        target = source3_lookup[
                            true_id
                        ]

                        print("\n")
                        print("=" * 60)
                        print(
                            f"MISSED S3 EXAMPLE "
                            f"#{shown_s3 + 1}"
                        )
                        print("=" * 60)

                        print(
                            "S1 name:",
                            s1["name"]
                        )

                        print(
                            "S3 name:",
                            target["name"]
                        )

                        print(
                            "S1 address:",
                            s1["address"]
                        )

                        print(
                            "S3 address:",
                            target["address"]
                        )

                        print(
                            "Common tokens:",
                            get_tokens(
                                s1["name"]
                            ) &
                            get_tokens(
                                target["name"]
                            )
                        )

                        shown_s3 += 1


# ============================================================
# 11. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("MISSED MATCH ANALYSIS COMPLETE")
print("=" * 60)

print(
    "\nTotal missed S2 true matches:",
    missed_s2
)

print(
    "Total missed S3 true matches:",
    missed_s3
)

print(
    "\nDisplayed S2 examples:",
    shown_s2
)

print(
    "Displayed S3 examples:",
    shown_s3
)

print("\nFinished.")