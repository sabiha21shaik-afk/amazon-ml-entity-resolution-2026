import duckdb
import os

TEST = "dataset/test"
OUTPUT = "output"

os.makedirs(OUTPUT, exist_ok=True)

S1 = f"{TEST}/test_source1.tsv"
S2 = f"{TEST}/test_source2.tsv"
S3 = f"{TEST}/test_source3.tsv"

MATCHING = f"{OUTPUT}/matching_results.tsv"
CANDIDATES = f"{OUTPUT}/candidate_pairs.tsv"

con = duckdb.connect()

print("=" * 60)
print("AMAZON BUSINESS ENTITY RESOLUTION")
print("DUCKDB FINAL PIPELINE")
print("=" * 60)

# ------------------------------------------------------------
# 1. Read the three test sources
# ------------------------------------------------------------

print("\nLoading test data...")

con.execute(f"""
CREATE OR REPLACE TABLE s1 AS
SELECT *
FROM read_csv(
    '{S1}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")

con.execute(f"""
CREATE OR REPLACE TABLE s2 AS
SELECT *
FROM read_csv(
    '{S2}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")

con.execute(f"""
CREATE OR REPLACE TABLE s3 AS
SELECT *
FROM read_csv(
    '{S3}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")

print("Test data loaded.")


# ------------------------------------------------------------
# 2. Create normalized columns
# ------------------------------------------------------------

print("\nNormalizing business names and addresses...")

for table in ["s1", "s2", "s3"]:

    con.execute(f"""
    ALTER TABLE {table}
    ADD COLUMN IF NOT EXISTS name_norm VARCHAR
    """)

    con.execute(f"""
    ALTER TABLE {table}
    ADD COLUMN IF NOT EXISTS address_norm VARCHAR
    """)

    con.execute(f"""
    UPDATE {table}
    SET
        name_norm = regexp_replace(
            lower(coalesce(business_name, '')),
            '[^a-z0-9]',
            '',
            'g'
        ),
        address_norm = regexp_replace(
            lower(coalesce(business_address, '')),
            '[^a-z0-9]',
            '',
            'g'
        )
    """)

print("Normalization completed.")


# ------------------------------------------------------------
# 3. Create candidate pairs using exact normalized name
# ------------------------------------------------------------

print("\nGenerating name-based candidates...")

con.execute("""
CREATE OR REPLACE TABLE name_candidates AS

SELECT
    a.entity_id AS source1_entity_id,
    b.entity_id AS candidate_entity_id
FROM s1 a
JOIN s2 b
    ON a.country = b.country
   AND a.name_norm = b.name_norm
   AND a.name_norm <> ''

UNION

SELECT
    a.entity_id AS source1_entity_id,
    b.entity_id AS candidate_entity_id
FROM s1 a
JOIN s3 b
    ON a.country = b.country
   AND a.name_norm = b.name_norm
   AND a.name_norm <> ''
""")

name_count = con.execute(
    "SELECT COUNT(*) FROM name_candidates"
).fetchone()[0]

print("Name candidates:", f"{name_count:,}")


# ------------------------------------------------------------
# 4. Create candidate pairs using exact normalized address
# ------------------------------------------------------------

print("\nGenerating address-based candidates...")

con.execute("""
CREATE OR REPLACE TABLE address_candidates AS

SELECT
    a.entity_id AS source1_entity_id,
    b.entity_id AS candidate_entity_id
FROM s1 a
JOIN s2 b
    ON a.country = b.country
   AND a.address_norm = b.address_norm
   AND a.address_norm <> ''

UNION

SELECT
    a.entity_id AS source1_entity_id,
    b.entity_id AS candidate_entity_id
FROM s1 a
JOIN s3 b
    ON a.country = b.country
   AND a.address_norm = b.address_norm
   AND a.address_norm <> ''
""")

address_count = con.execute(
    "SELECT COUNT(*) FROM address_candidates"
).fetchone()[0]

print("Address candidates:", f"{address_count:,}")


# ------------------------------------------------------------
# 5. Combine candidates
# ------------------------------------------------------------

print("\nCombining candidates...")

con.execute("""
CREATE OR REPLACE TABLE candidates AS

SELECT source1_entity_id, candidate_entity_id
FROM name_candidates

UNION

SELECT source1_entity_id, candidate_entity_id
FROM address_candidates
""")

candidate_count = con.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

print("Total candidate pairs:", f"{candidate_count:,}")


# ------------------------------------------------------------
# 6. Save candidate_pairs.tsv
# ------------------------------------------------------------

print("\nWriting candidate_pairs.tsv...")

con.execute(f"""
COPY candidates
TO '{CANDIDATES}'
(
    FORMAT CSV,
    DELIMITER '\\t',
    HEADER
)
""")

print("candidate_pairs.tsv created.")


# ------------------------------------------------------------
# 7. Create matching results
# ------------------------------------------------------------

print("\nCreating matching_results.tsv...")

con.execute("""
CREATE OR REPLACE TABLE matching_results AS

SELECT
    s.entity_id AS source1_entity_id,

    COALESCE(
        string_agg(
            c.candidate_entity_id,
            ','
            ORDER BY c.candidate_entity_id
        ),
        ''
    ) AS matched_entity_ids

FROM s1 s

LEFT JOIN candidates c
    ON s.entity_id = c.source1_entity_id

GROUP BY s.entity_id
ORDER BY s.entity_id
""")

con.execute(f"""
COPY matching_results
TO '{MATCHING}'
(
    FORMAT CSV,
    DELIMITER '\\t',
    HEADER
)
""")

print("matching_results.tsv created.")


# ------------------------------------------------------------
# 8. Validation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

s1_count = con.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

result_count = con.execute(
    "SELECT COUNT(*) FROM matching_results"
).fetchone()[0]

candidate_count = con.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

matched_count = con.execute("""
SELECT COUNT(*)
FROM matching_results
WHERE matched_entity_ids <> ''
""").fetchone()[0]

print("Test Source 1 rows :", f"{s1_count:,}")
print("Result rows        :", f"{result_count:,}")
print("Matched S1 rows    :", f"{matched_count:,}")
print("Candidate pairs    :", f"{candidate_count:,}")

print("\nFiles created:")
print(MATCHING)
print(CANDIDATES)

print("\n" + "=" * 60)
print("PIPELINE COMPLETED")
print("=" * 60)