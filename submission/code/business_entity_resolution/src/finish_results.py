import duckdb
import os

TEST = "dataset/test"
OUTPUT = "output"

S1 = f"{TEST}/test_source1.tsv"
CANDIDATES = f"{OUTPUT}/candidate_pairs.tsv"
RESULT = f"{OUTPUT}/matching_results.tsv"

con = duckdb.connect()

print("=" * 60)
print("CREATING FINAL MATCHING RESULTS")
print("=" * 60)

print("\nReading Source 1 and candidate pairs...")

query = f"""
COPY (
    SELECT
        s.entity_id AS source1_entity_id,
        COALESCE(
            string_agg(c.candidate_entity_id, ','),
            ''
        ) AS matched_entity_ids
    FROM read_csv(
        '{S1}',
        delim='\\t',
        header=true,
        all_varchar=true
    ) s
    LEFT JOIN read_csv(
        '{CANDIDATES}',
        delim='\\t',
        header=true,
        all_varchar=true
    ) c
    ON s.entity_id = c.source1_entity_id
    GROUP BY s.entity_id
)
TO '{RESULT}'
(
    FORMAT CSV,
    DELIMITER '\\t',
    HEADER
)
"""

print("\nGenerating matching_results.tsv...")
print("This is the final aggregation step.")

con.execute(query)

print("\n" + "=" * 60)
print("MATCHING RESULTS CREATED")
print("=" * 60)

print(f"\nFile: {RESULT}")

count = con.execute(f"""
SELECT COUNT(*)
FROM read_csv(
    '{RESULT}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""").fetchone()[0]

print("Rows:", f"{count:,}")

print("\nDONE")