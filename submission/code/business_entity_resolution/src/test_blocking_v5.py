import pandas as pd
import re
from collections import defaultdict, Counter


# ============================================================
# 1. NORMALIZE BUSINESS NAME
# ============================================================

def normalize_name(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9]",
        "",
        text
    )

    return text


# ============================================================
# 2. GET NAME TOKENS
# ============================================================

def get_tokens(text):

    if pd.isna(text):
        return []

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    tokens = text.split()

    # Ignore extremely short words
    tokens = [
        token
        for token in tokens
        if len(token) >= 4
    ]

    return set(tokens)


# ============================================================
# 3. CREATE NAME SIGNATURES
# ============================================================

def create_signatures(name):

    name = normalize_name(name)

    if not name:
        return []

    signatures = set()

    # Full name
    signatures.add(
        "full:" + name
    )

    # First 6 characters
    signatures.add(
        "prefix6:" + name[:6]
    )

    # First 8 characters
    signatures.add(
        "prefix8:" + name[:8]
    )

    # First 4 + last 4
    if len(name) >= 8:

        signatures.add(
            "edge:" +
            name[:4] +
            name[-4:]
        )

    # First 5 + last 3
    if len(name) >= 8:

        signatures.add(
            "edge2:" +
            name[:5] +
            name[-3:]
        )

    return signatures


# ============================================================
# 4. LOAD DATA
# ============================================================

print("Loading datasets...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Datasets loaded.")


# ============================================================
# 5. SOURCE 1 FAST LOOKUP
# ============================================================

print("\nCreating Source 1 lookup...")

source1_lookup = {}

for row in source1.itertuples(index=False):

    source1_lookup[row.entity_id] = (
        row.business_name,
        row.country
    )

print("Source 1 lookup created.")


# ============================================================
# 6. BUILD NAME SIGNATURE INDEX
# ============================================================

def build_name_index(df):

    index = defaultdict(set)

    for row in df.itertuples(index=False):

        country = str(row.country)

        signatures = create_signatures(
            row.business_name
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

    return index


# ============================================================
# 7. BUILD TOKEN FREQUENCY
# ============================================================

def build_token_frequency(df):

    frequency = Counter()

    for row in df.itertuples(index=False):

        tokens = get_tokens(
            row.business_name
        )

        for token in tokens:

            frequency[token] += 1

    return frequency


# ============================================================
# 8. BUILD TOKEN INDEX
# ============================================================

MAX_TOKEN_FREQUENCY = 1000


def build_token_index(df, frequency):

    index = defaultdict(set)

    for row in df.itertuples(index=False):

        country = str(row.country)

        tokens = get_tokens(
            row.business_name
        )

        for token in tokens:

            if frequency[token] <= MAX_TOKEN_FREQUENCY:

                key = (
                    country +
                    "|" +
                    token
                )

                index[key].add(
                    row.entity_id
                )

    return index


# ============================================================
# 9. SOURCE 2 INDEXES
# ============================================================

print("\nBuilding Source 2 name index...")

name_index_s2 = build_name_index(
    source2
)

print("Source 2 name index created.")

print("Counting Source 2 tokens...")

frequency_s2 = build_token_frequency(
    source2
)

print("Building Source 2 token index...")

token_index_s2 = build_token_index(
    source2,
    frequency_s2
)

print("Source 2 token index created.")


# ============================================================
# 10. SOURCE 3 INDEXES
# ============================================================

print("\nBuilding Source 3 name index...")

name_index_s3 = build_name_index(
    source3
)

print("Source 3 name index created.")

print("Counting Source 3 tokens...")

frequency_s3 = build_token_frequency(
    source3
)

print("Building Source 3 token index...")

token_index_s3 = build_token_index(
    source3,
    frequency_s3
)

print("Source 3 token index created.")


# ============================================================
# 11. CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    business_name,
    country,
    name_index,
    token_index,
    token_frequency
):

    country = str(country)

    candidates = set()

    # --------------------------------------------------------
    # NAME SIGNATURE BLOCKS
    # --------------------------------------------------------

    signatures = create_signatures(
        business_name
    )

    signature_blocks = []

    for signature in signatures:

        key = (
            country +
            "|" +
            signature
        )

        if key in name_index:

            block_size = len(
                name_index[key]
            )

            signature_blocks.append(
                (
                    block_size,
                    key
                )
            )

    # Sort smallest blocks first
    signature_blocks.sort(
        key=lambda x: x[0]
    )

    # Take the 2 smallest useful name blocks
    selected_name_blocks = signature_blocks[:2]

    for block_size, key in selected_name_blocks:

        candidates.update(
            name_index[key]
        )


    # --------------------------------------------------------
    # RARE TOKEN BLOCKS
    # --------------------------------------------------------

    tokens = get_tokens(
        business_name
    )

    token_blocks = []

    for token in tokens:

        frequency = token_frequency.get(
            token,
            999999999
        )

        if frequency <= MAX_TOKEN_FREQUENCY:

            key = (
                country +
                "|" +
                token
            )

            if key in token_index:

                block_size = len(
                    token_index[key]
                )

                token_blocks.append(
                    (
                        frequency,
                        block_size,
                        key
                    )
                )

    # Rarest tokens first
    token_blocks.sort(
        key=lambda x: x[0]
    )

    # Take 2 rarest tokens
    selected_token_blocks = token_blocks[:2]

    for frequency, block_size, key in selected_token_blocks:

        candidates.update(
            token_index[key]
        )

    return candidates


# ============================================================
# 12. INITIALIZE RESULTS
# ============================================================

s2_true = 0
s2_captured = 0

s3_true = 0
s3_captured = 0

s2_candidate_total = 0
s3_candidate_total = 0

s2_candidate_max = 0
s3_candidate_max = 0


# ============================================================
# 13. PROCESS GROUND TRUTH
# ============================================================

print("\nTesting combined blocking...")

total_rows = len(
    ground_truth
)


for counter, row in enumerate(
    ground_truth.itertuples(index=False),
    start=1
):

    s1_id = row.source1_entity_id

    if s1_id not in source1_lookup:
        continue

    business_name, country = (
        source1_lookup[s1_id]
    )


    # ========================================================
    # SOURCE 2 CANDIDATES
    # ========================================================

    candidates_s2 = generate_candidates(
        business_name,
        country,
        name_index_s2,
        token_index_s2,
        frequency_s2
    )

    count_s2 = len(
        candidates_s2
    )

    s2_candidate_total += count_s2

    if count_s2 > s2_candidate_max:

        s2_candidate_max = count_s2


    # ========================================================
    # SOURCE 3 CANDIDATES
    # ========================================================

    candidates_s3 = generate_candidates(
        business_name,
        country,
        name_index_s3,
        token_index_s3,
        frequency_s3
    )

    count_s3 = len(
        candidates_s3
    )

    s3_candidate_total += count_s3

    if count_s3 > s3_candidate_max:

        s3_candidate_max = count_s3


    # ========================================================
    # GROUND TRUTH
    # ========================================================

    matched_ids = row.matched_entity_ids

    if pd.isna(matched_ids):
        continue

    true_ids = str(
        matched_ids
    ).split(",")


    # ========================================================
    # SOURCE 2 TRUE MATCHES
    # ========================================================

    true_s2_ids = [
        x
        for x in true_ids
        if x.startswith("S2-")
    ]

    for true_id in true_s2_ids:

        s2_true += 1

        if true_id in candidates_s2:

            s2_captured += 1


    # ========================================================
    # SOURCE 3 TRUE MATCHES
    # ========================================================

    true_s3_ids = [
        x
        for x in true_ids
        if x.startswith("S3-")
    ]

    for true_id in true_s3_ids:

        s3_true += 1

        if true_id in candidates_s3:

            s3_captured += 1


    # ========================================================
    # PROGRESS
    # ========================================================

    if counter % 100000 == 0:

        print(
            f"Processed {counter:,} / "
            f"{total_rows:,} Source 1 records"
        )


# ============================================================
# 14. CALCULATE RECALL
# ============================================================

s2_recall = (
    s2_captured /
    s2_true *
    100
    if s2_true > 0
    else 0
)

s3_recall = (
    s3_captured /
    s3_true *
    100
    if s3_true > 0
    else 0
)


average_s2_candidates = (
    s2_candidate_total /
    len(source1)
)

average_s3_candidates = (
    s3_candidate_total /
    len(source1)
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("BLOCKING V5 RESULTS")
print("=" * 60)


print("\nSOURCE 2")
print("-" * 40)

print(
    "True matches:",
    s2_true
)

print(
    "Captured:",
    s2_captured
)

print(
    "Recall: {:.2f}%".format(
        s2_recall
    )
)

print(
    "Average candidates per Source 1: {:.2f}".format(
        average_s2_candidates
    )
)

print(
    "Maximum candidates for one Source 1:",
    s2_candidate_max
)


print("\nSOURCE 3")
print("-" * 40)

print(
    "True matches:",
    s3_true
)

print(
    "Captured:",
    s3_captured
)

print(
    "Recall: {:.2f}%".format(
        s3_recall
    )
)

print(
    "Average candidates per Source 1: {:.2f}".format(
        average_s3_candidates
    )
)

print(
    "Maximum candidates for one Source 1:",
    s3_candidate_max
)


print("\n")
print("=" * 60)
print("BLOCKING V5 TEST COMPLETED")
print("=" * 60)