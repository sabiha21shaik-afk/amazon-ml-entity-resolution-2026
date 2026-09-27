# Amazon ML Challenge 2026
# Business Entity Resolution

## 1. Project Overview

This project solves the Business Entity Resolution challenge by identifying matching business entities across multiple data sources.

The task is to match businesses from Source 1 with corresponding entities in Source 2 and Source 3.

The solution focuses on scalable candidate generation using business names and addresses.

---

## 2. Dataset

The dataset contains three source files:

- Source 1
- Source 2
- Source 3

Each business record contains:

- entity_id
- business_name
- business_address
- country

The training data also provides ground-truth mappings between Source 1 and the matching entities in Source 2 and Source 3.

---

## 3. Data Preprocessing

Business names and addresses were normalized before matching.

The normalization process includes:

- Converting text to lowercase
- Removing non-alphanumeric characters
- Creating normalized business-name representations
- Extracting useful address tokens

This helps identify businesses even when formatting differs between sources.

---

## 4. Candidate Generation / Blocking

Because the datasets contain millions of records, comparing every Source 1 record with every Source 2 and Source 3 record would be computationally expensive.

Therefore, blocking was used to generate a smaller set of possible candidates.

The blocking approach uses:

- Exact normalized business-name blocks
- Business-name prefixes
- Business-name edge signatures
- Address tokens
- Adjacent address-token pairs
- Country information

Multiple blocking strategies were combined to improve candidate recall.

---

## 5. Matching Approach

Candidate pairs were generated using the blocking strategy.

The generated candidate pairs were then used to produce the final matching results.

The final output contains:

- `source1_entity_id`
- `matched_entity_ids`

Only candidate entities from Source 2 and Source 3 are considered for matching.

---

## 6. Candidate Output

The final candidate file contains:

- `source1_entity_id`
- `candidate_entity_ids`

Candidate IDs are stored as comma-separated values for each Source 1 entity.

This allows the final matching results to be checked against the generated candidate set.

---

## 7. Output Files

The final submission contains:

### matching_results.tsv

Contains the predicted matching entities for every Source 1 entity.

### candidate_pairs.tsv

Contains the candidate entities generated for every Source 1 entity.

---

## 8. Tools and Technologies

The project was implemented using:

- Python
- DuckDB
- Pandas
- Regular expressions
- TSV data processing

DuckDB was used to process the large candidate-pair dataset efficiently.

---

## 9. Scalability

The dataset contains millions of records across the three sources.

To avoid an all-pairs comparison, the solution uses blocking and candidate generation.

This significantly reduces the search space before producing the final matching results.

---

## 10. Project Structure

```text
student_resource/
│
├── dataset/
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── utils/
│
├── explore_dataset.py
├── analyze_matching.py
├── test_blocking.py
├── evaluate_blocking.py
├── test_blocking_v2.py
├── test_blocking_v3.py
├── test_blocking_v4.py
├── test_blocking_v5.py
├── test_blocking_v6.py
├── test_blocking_v7.py
├── test_blocking_v8.py
├── build_final_pipeline.py
│
└── Documentation_template.md