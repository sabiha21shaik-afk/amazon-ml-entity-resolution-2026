import os
import pandas as pd

dataset_folder = "dataset"

print("\n" + "=" * 60)
print("DATASET FILE STRUCTURE")
print("=" * 60)

# Show all files and folders
for root, folders, files in os.walk(dataset_folder):

    print(f"\nFolder: {root}")

    for file in files:
        print(f"  - {file}")


print("\n" + "=" * 60)
print("TSV DATASET EXPLORATION")
print("=" * 60)


# Find TSV files inside all subfolders
for root, folders, files in os.walk(dataset_folder):

    for file in files:

        if file.lower().endswith(".tsv"):

            file_path = os.path.join(root, file)

            print("\n" + "=" * 60)
            print(f"READING FILE: {file_path}")
            print("=" * 60)

            # Read TSV file
            df = pd.read_csv(file_path, sep="\t")

            print("\nShape:")
            print(df.shape)

            print("\nColumns:")
            print(df.columns.tolist())

            print("\nFirst 5 rows:")
            print(df.head())

            print("\nData types:")
            print(df.dtypes)

            print("\nMissing values:")
            print(df.isnull().sum())

            print("\nDuplicate rows:")
            print(df.duplicated().sum())