import os
from merlin.datasets.synthetic import generate_data

DATA_DIR = "/workspace/project/data/"

mock_train, mock_valid = generate_data("aliccp-raw", num_rows=100000, set_sizes=(0.7, 0.3))

# Save payloads down into designated raw partitions
os.makedirs(os.path.join(DATA_DIR, "raw/train"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "raw/valid"), exist_ok=True)

mock_train.to_ddf().to_parquet(os.path.join(DATA_DIR, "raw/train/"))
mock_valid.to_ddf().to_parquet(os.path.join(DATA_DIR, "raw/valid/"))

print("Successfully seeded raw parquet interaction logs!")
