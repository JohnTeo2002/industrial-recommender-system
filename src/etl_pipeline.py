import os

# Completely disable TensorFlow's GPU access to stop the framework collision
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import nvtabular as nvt
from nvtabular.ops import Categorify, FillMissing, AddTags
from merlin.schema.tags import Tags
from merlin.io.dataset import Dataset

'''
This code uses NVTabular (NVIDIA's GPU-accelerated data processing library) to perform ETL (Extract, Transform, Load) on recommendation system data. 
The goal is to take raw user-item interaction data and transform it into a format ready for training a machine learning model, particularly for click-through rate (CTR) prediction.
In production recommendation systems, there are billions of interactions, hundreds of features and models need fresh data constantly.
NVTabular solves this by using GPU acceleration to process massive datasets 10-100x faster than traditional CPU-based tools.
'''

def run_production_etl(data_dir="/workspace/project/data/"):
    print("Initializing GPU-Accelerated NVTabular ETL Engine...")
    
    # 1. Define Input Target Folders
    train_path = os.path.join(data_dir, "raw/train/")
    valid_path = os.path.join(data_dir, "raw/valid/")
    
    # 2. Instantiate Streaming Merlin Datasets
    # Part_size tells NVTabular how to chunk memory to fit the GPU VRAM cleanly
    train_ds = Dataset(train_path, engine="parquet", part_size="100MB")
    valid_ds = Dataset(valid_path, engine="parquet", part_size="100MB")
    
    # 3. Architect the Accelerated Feature Computation Graph
    # We tag features explicitly so down-stream Merlin models know how to parse them
    user_id = ["user_id"] >> Categorify() >> AddTags(tags=[Tags.USER_ID, Tags.CATEGORICAL, Tags.USER])
    item_id = ["item_id"] >> Categorify() >> AddTags(tags=[Tags.ITEM_ID, Tags.CATEGORICAL, Tags.ITEM])
    
    # Process additional context features (e.g., categories)
    user_features = ["user_position", "user_age"] >> Categorify() >> AddTags(tags=[Tags.USER, Tags.CATEGORICAL])
    item_features = ["item_category"] >> Categorify() >> AddTags(tags=[Tags.ITEM, Tags.CATEGORICAL])
    
    # Process target labels for click-through rate scoring loops
    targets = ["click"] >> AddTags(tags=[Tags.TARGET, Tags.BINARY_CLASSIFICATION])
    
    # Group the computational node graph chains together
    pipeline_graph = user_id + item_id + user_features + item_features + targets
    
    # 4. Compile the Workflow Instance
    workflow = nvt.Workflow(pipeline_graph)
    
    # 5. Fit the Workflow & Execute GPU Transform Operations
    print("Fitting computational graph statistics on training split...")
    workflow.fit(train_ds)
    
    # Define production-grade target outputs
    output_train_dir = os.path.join(data_dir, "processed/train")
    output_valid_dir = os.path.join(data_dir, "processed/valid")
    
    print("Executing transformations and saving downstream Parquet shards...")
    workflow.transform(train_ds).to_parquet(output_path=output_train_dir)
    workflow.transform(valid_ds).to_parquet(output_path=output_valid_dir)
    
    # 6. Export the Compiled Schema and Workflow Metadata State
    # This schema is critical for automated MLOps and serving layers
    workflow.save(os.path.join(data_dir, "processed/workflow_etl"))
    print("Successfully exported NVTabular serialized execution graph.")
    print("Successfully generated unified Protobuf schema metadata file (schema.pbtxt).")

if __name__ == "__main__":
    run_production_etl()
