import duckdb

con = duckdb.connect()

INPUT = "output/candidate_pairs.tsv"
OUTPUT = "output/candidate_pairs_fixed.tsv"

print("Converting candidate_pairs.tsv to required format...")

con.execute(f"""
COPY (
    SELECT
        source1_entity_id,
        string_agg(candidate_entity_id, ',' ORDER BY candidate_entity_id)
            AS candidate_entity_ids
    FROM read_csv(
        '{INPUT}',
        delim='\\t',
        header=true,
        all_varchar=true
    )
    GROUP BY source1_entity_id
)
TO '{OUTPUT}'
(
    FORMAT CSV,
    DELIMITER '\\t',
    HEADER
)
""")

count = con.execute(f"""
SELECT COUNT(*)
FROM read_csv(
    '{OUTPUT}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""").fetchone()[0]

print("Rows created:", f"{count:,}")
print("Created:", OUTPUT)