import pandas as pd
import re
from collections import defaultdict, Counter
import time


# ============================================================
# SETTINGS
# ============================================================

# Blocks larger than this are ignored.
# This prevents extremely common words/cities from creating
# hundreds of thousands of candidates.
MAX_NAME_BLOCK_SIZE = 50000
MAX_ADDRESS_BLOCK_SIZE = 20000

# Address tokens appearing more than this many times are ignored.
MAX_ADDRESS_TOKEN_FREQ = 5000

# Address pairs appearing more than this many times are ignored.
MAX_ADDRESS_PAIR_FREQ = 5000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep only letters and numbers.
    text = re.sub(r"[^a-z0-9]+", "", text)

    return text


def address_tokens(text):
    if pd.isna(text):
        return []

    text = str(text).lower()

    # Split address into useful pieces.
    parts = re.findall(r"[a-z0-9]+", text)

    # Keep reasonably informative tokens.
    return [x for x in parts if len(x) >= 3]


# ============================================================
# NAME BLOCK SIGNATURES
# ============================================================

def name_signatures(name):
    name = normalize(name)

    if not name:
        return []

    signatures = set()

    # Full normalized name
    signatures.add("full:" + name)

    # First 6 characters
    if len(name) >= 6:
        signatures.add("prefix6:" + name[:6])

    # First 8 characters
    if len(name) >= 8:
        signatures.add("prefix8:" + name[:8])

    # First 4 + last 4
    if len(name) >= 8:
        signatures.add("edge:" + name[:4] + name[-4:])

    # First 5 + last 3
    if len(name) >= 8:
        signatures.add("edge2:" + name[:5] + name[-3:])

    return list(signatures)


# ============================================================
# ADDRESS BLOCK SIGNATURES
# ============================================================

def address_signatures(address):
    tokens = address_tokens(address)

    signatures = set()

    # Individual address tokens
    for token in tokens:
        signatures.add("token:" + token)

    # Adjacent token pairs
    for i in range(len(tokens) - 1):
        pair = tokens[i] + "|" + tokens[i + 1]
        signatures.add("pair:" + pair)

    return list(signatures)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

train_s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

train_s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Source1:", len(train_s1))
print("Source2:", len(train_s2))
print("Source3:", len(train_s3))


# ============================================================
# BUILD ADDRESS FREQUENCY COUNTS
# ============================================================

print("\nBuilding address token frequencies...")

def build_address_frequency(df):

    counter = Counter()

    for address in df["business_address"]:

        tokens = set(address_tokens(address))

        for token in tokens:
            counter[token] += 1

    return counter


s2_address_freq = build_address_frequency(train_s2)
s3_address_freq = build_address_frequency(train_s3)

print("S2 address tokens:", len(s2_address_freq))
print("S3 address tokens:", len(s3_address_freq))


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

def build_index(df, address_freq):

    index = defaultdict(list)

    for idx, row in df.iterrows():

        name = row["business_name"]
        address = row["business_address"]

        # ----------------------------
        # NAME BLOCKS
        # ----------------------------

        for sig in name_signatures(name):

            index[sig].append(idx)

        # ----------------------------
        # ADDRESS BLOCKS
        # ----------------------------

        tokens = address_tokens(address)

        # Individual tokens
        for token in set(tokens):

            freq = address_freq[token]

            # Ignore extremely common address tokens
            if freq > MAX_ADDRESS_TOKEN_FREQ:
                continue

            sig = "token:" + token

            index[sig].append(idx)

        # Adjacent pairs
        for i in range(len(tokens) - 1):

            pair = tokens[i] + "|" + tokens[i + 1]

            sig = "pair:" + pair

            index[sig].append(idx)

    return index


print("\nBuilding S2 index...")
s2_index = build_index(train_s2, s2_address_freq)

print("Building S3 index...")
s3_index = build_index(train_s3, s3_address_freq)


# ============================================================
# REMOVE OVERSIZED BLOCKS
# ============================================================

def filter_index(index):

    filtered = {}

    for sig, values in index.items():

        size = len(values)

        if sig.startswith("token:"):

            if size > MAX_ADDRESS_BLOCK_SIZE:
                continue

        elif sig.startswith("pair:"):

            if size > MAX_ADDRESS_BLOCK_SIZE:
                continue

        else:
            # Name block
            if size > MAX_NAME_BLOCK_SIZE:
                continue

        filtered[sig] = values

    return filtered


print("\nFiltering oversized blocks...")

s2_index = filter_index(s2_index)
s3_index = filter_index(s3_index)

print("S2 usable blocks:", len(s2_index))
print("S3 usable blocks:", len(s3_index))


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(row, index):

    candidates = set()

    # ---------------------------------------
    # NAME BLOCKS
    # ---------------------------------------

    name_blocks = []

    for sig in name_signatures(row["business_name"]):

        if sig in index:

            name_blocks.append(
                (len(index[sig]), sig)
            )

    # Use the 2 smallest name blocks.
    name_blocks.sort()

    for size, sig in name_blocks[:2]:

        for idx in index[sig]:

            candidates.add(idx)

    # ---------------------------------------
    # ADDRESS BLOCKS
    # ---------------------------------------

    address_blocks = []

    for sig in address_signatures(row["business_address"]):

        if sig in index:

            address_blocks.append(
                (len(index[sig]), sig)
            )

    # Use the 3 smallest address blocks.
    address_blocks.sort()

    for size, sig in address_blocks[:3]:

        for idx in index[sig]:

            candidates.add(idx)

    return candidates


# ============================================================
# EVALUATE BLOCKING
# ============================================================

print("\nEvaluating V8 blocking...")

# ------------------------------------------------------------
# Build ground truth lookup
# ------------------------------------------------------------

truth_s2 = defaultdict(set)
truth_s3 = defaultdict(set)

print("\nReading ground truth...")

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]):
        continue

    matched = str(row["matched_entity_ids"])

    ids = matched.split(",")

    for entity_id in ids:

        entity_id = entity_id.strip()

        if entity_id.startswith("source2_"):

            truth_s2[s1_id].add(entity_id)

        elif entity_id.startswith("source3_"):

            truth_s3[s1_id].add(entity_id)


# ============================================================
# ENTITY ID -> DATAFRAME INDEX
# ============================================================

s2_id_to_index = {
    entity_id: idx
    for idx, entity_id in enumerate(train_s2["entity_id"])
}

s3_id_to_index = {
    entity_id: idx
    for idx, entity_id in enumerate(train_s3["entity_id"])
}


# ============================================================
# PROCESS SOURCE 1
# ============================================================

total_true_s2 = 0
captured_s2 = 0

total_true_s3 = 0
captured_s3 = 0

total_candidates_s2 = 0
total_candidates_s3 = 0

max_candidates_s2 = 0
max_candidates_s3 = 0

start = time.time()


for i, row in train_s1.iterrows():

    s1_id = row["entity_id"]

    # ---------------------------------------
    # S2 candidates
    # ---------------------------------------

    candidates_s2 = generate_candidates(row, s2_index)

    total_candidates_s2 += len(candidates_s2)

    if len(candidates_s2) > max_candidates_s2:
        max_candidates_s2 = len(candidates_s2)

    true_s2_ids = truth_s2.get(s1_id, set())

    total_true_s2 += len(true_s2_ids)

    for true_id in true_s2_ids:

        true_idx = s2_id_to_index.get(true_id)

        if true_idx in candidates_s2:

            captured_s2 += 1

    # ---------------------------------------
    # S3 candidates
    # ---------------------------------------

    candidates_s3 = generate_candidates(row, s3_index)

    total_candidates_s3 += len(candidates_s3)

    if len(candidates_s3) > max_candidates_s3:
        max_candidates_s3 = len(candidates_s3)

    true_s3_ids = truth_s3.get(s1_id, set())

    total_true_s3 += len(true_s3_ids)

    for true_id in true_s3_ids:

        true_idx = s3_id_to_index.get(true_id)

        if true_idx in candidates_s3:

            captured_s3 += 1

    # ---------------------------------------
    # Progress
    # ---------------------------------------

    if (i + 1) % 100000 == 0:

        elapsed = time.time() - start

        print(
            f"Processed {i + 1:,} / {len(train_s1):,} "
            f"({elapsed:.1f}s)"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("V8 BLOCKING RESULTS")
print("=" * 60)

print("\nSOURCE 2")

print("True matches:", f"{total_true_s2:,}")
print("Captured:", f"{captured_s2:,}")

recall_s2 = (
    captured_s2 / total_true_s2
    if total_true_s2
    else 0
)

print("Recall:", f"{recall_s2 * 100:.2f}%")

avg_s2 = total_candidates_s2 / len(train_s1)

print("Total candidates:", f"{total_candidates_s2:,}")
print("Average candidates per S1:", f"{avg_s2:.2f}")
print("Maximum candidates:", f"{max_candidates_s2:,}")


print("\nSOURCE 3")

print("True matches:", f"{total_true_s3:,}")
print("Captured:", f"{captured_s3:,}")

recall_s3 = (
    captured_s3 / total_true_s3
    if total_true_s3
    else 0
)

print("Recall:", f"{recall_s3 * 100:.2f}%")

avg_s3 = total_candidates_s3 / len(train_s1)

print("Total candidates:", f"{total_candidates_s3:,}")
print("Average candidates per S1:", f"{avg_s3:.2f}")
print("Maximum candidates:", f"{max_candidates_s3:,}")


# ============================================================
# OVERALL
# ============================================================

total_true = total_true_s2 + total_true_s3
total_captured = captured_s2 + captured_s3

overall_recall = (
    total_captured / total_true
    if total_true
    else 0
)

total_candidates = total_candidates_s2 + total_candidates_s3

avg_total = total_candidates / len(train_s1)

max_total = max_candidates_s2 + max_candidates_s3


print("\n")
print("=" * 60)
print("OVERALL V8 RESULTS")
print("=" * 60)

print("Total true matches:", f"{total_true:,}")
print("Total captured:", f"{total_captured:,}")

print(
    "Overall recall:",
    f"{overall_recall * 100:.2f}%"
)

print(
    "Total candidates:",
    f"{total_candidates:,}"
)

print(
    "Average candidates per S1:",
    f"{avg_total:.2f}"
)

print(
    "Maximum combined candidates:",
    f"{max_total:,}"
)

print("=" * 60)

print("\nV8 evaluation completed.")