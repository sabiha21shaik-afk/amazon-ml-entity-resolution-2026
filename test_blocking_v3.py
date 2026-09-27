import pandas as pd
import re
from collections import defaultdict


# ==========================================
# 1. NORMALIZE BUSINESS NAME
# ==========================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", "", text)

    return text


# ==========================================
# 2. CREATE MULTIPLE NAME SIGNATURES
# ==========================================

def create_signatures(name):

    name = normalize_text(name)

    if not name:
        return []

    signatures = set()

    # Full normalized name
    signatures.add("full:" + name)

    # First 6 characters
    signatures.add("prefix6:" + name[:6])

    # First 8 characters
    signatures.add("prefix8:" + name[:8])

    # First 4 + last 4
    if len(name) >= 8:
        signatures.add(
            "edge:" + name[:4] + name[-4:]
        )

    # First 5 + last 3
    if len(name) >= 8:
        signatures.add(
            "edge2:" + name[:5] + name[-3:]
        )

    return signatures


# ==========================================
# 3. LOAD DATA
# ==========================================

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


# ==========================================
# 4. CREATE FAST SOURCE 1 LOOKUP
# ==========================================

print("\nCreating Source 1 lookup...")

source1_lookup = {}

for row in source1.itertuples(index=False):

    source1_lookup[row.entity_id] = (
        row.business_name,
        row.country
    )

print("Source 1 lookup created.")


# ==========================================
# 5. BUILD BLOCKING INDEX
# ==========================================

def build_index(df):

    index = defaultdict(set)

    for row in df.itertuples(index=False):

        country = str(row.country)

        signatures = create_signatures(
            row.business_name
        )

        for signature in signatures:

            key = country + "|" + signature

            index[key].add(row.entity_id)

    return index


# ==========================================
# 6. BUILD SOURCE 2 INDEX
# ==========================================

print("\nBuilding Source 2 index...")

index_s2 = build_index(source2)

print("Source 2 index created.")


# ==========================================
# 7. BUILD SOURCE 3 INDEX
# ==========================================

print("Building Source 3 index...")

index_s3 = build_index(source3)

print("Source 3 index created.")


# ==========================================
# 8. INITIALIZE COUNTERS
# ==========================================

s2_true = 0
s2_captured = 0

s3_true = 0
s3_captured = 0

s2_candidate_total = 0
s3_candidate_total = 0

s2_candidate_max = 0
s3_candidate_max = 0


# ==========================================
# 9. PROCESS GROUND TRUTH
# ==========================================

print("\nTesting blocking against ground truth...")

total_rows = len(ground_truth)

for counter, row in enumerate(
    ground_truth.itertuples(index=False),
    start=1
):

    s1_id = row.source1_entity_id

    # Fast dictionary lookup
    if s1_id not in source1_lookup:
        continue

    business_name, country = source1_lookup[s1_id]

    country = str(country)

    signatures = create_signatures(
        business_name
    )


    # ======================================
    # SOURCE 2 CANDIDATES
    # ======================================

    candidates_s2 = set()

    for signature in signatures:

        key = country + "|" + signature

        if key in index_s2:
            candidates_s2.update(
                index_s2[key]
            )

    candidate_count_s2 = len(candidates_s2)

    s2_candidate_total += candidate_count_s2

    if candidate_count_s2 > s2_candidate_max:
        s2_candidate_max = candidate_count_s2


    # ======================================
    # SOURCE 3 CANDIDATES
    # ======================================

    candidates_s3 = set()

    for signature in signatures:

        key = country + "|" + signature

        if key in index_s3:
            candidates_s3.update(
                index_s3[key]
            )

    candidate_count_s3 = len(candidates_s3)

    s3_candidate_total += candidate_count_s3

    if candidate_count_s3 > s3_candidate_max:
        s3_candidate_max = candidate_count_s3


    # ======================================
    # GROUND TRUTH
    # ======================================

    matched_ids = row.matched_entity_ids

    if pd.isna(matched_ids):
        continue

    true_ids = str(matched_ids).split(",")


    # ======================================
    # SOURCE 2 TRUE MATCHES
    # ======================================

    true_s2_ids = [
        x for x in true_ids
        if x.startswith("S2-")
    ]

    for true_id in true_s2_ids:

        s2_true += 1

        if true_id in candidates_s2:
            s2_captured += 1


    # ======================================
    # SOURCE 3 TRUE MATCHES
    # ======================================

    true_s3_ids = [
        x for x in true_ids
        if x.startswith("S3-")
    ]

    for true_id in true_s3_ids:

        s3_true += 1

        if true_id in candidates_s3:
            s3_captured += 1


    # ======================================
    # PROGRESS
    # ======================================

    if counter % 100000 == 0:

        print(
            f"Processed {counter:,} / "
            f"{total_rows:,} Source 1 records"
        )


# ==========================================
# 10. CALCULATE RECALL
# ==========================================

s2_recall = (
    s2_captured / s2_true * 100
    if s2_true > 0
    else 0
)

s3_recall = (
    s3_captured / s3_true * 100
    if s3_true > 0
    else 0
)


average_s2_candidates = (
    s2_candidate_total / len(source1)
)

average_s3_candidates = (
    s3_candidate_total / len(source1)
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n")
print("=" * 60)
print("BLOCKING V3 RESULTS")
print("=" * 60)


print("\nSOURCE 2")
print("-" * 40)

print("True matches:", s2_true)

print("Captured:", s2_captured)

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

print("True matches:", s3_true)

print("Captured:", s3_captured)

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
print("BLOCKING V3 TEST COMPLETED")
print("=" * 60)