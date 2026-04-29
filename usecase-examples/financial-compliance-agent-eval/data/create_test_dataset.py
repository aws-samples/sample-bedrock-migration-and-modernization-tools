#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Script to create test dataset from synthetic examples.
"""

import boto3
import json
import sagemaker
import os

def load_json_data():
    """Load the ground truth JSON data"""
    json_path = os.path.join(os.path.dirname(__file__), 'ground_truth.json')
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Take first 200 samples
        data = data[:20]
        print(f"Loaded ground truth data from JSON: {len(data)} samples")
        return data
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return None

def save_to_s3(data, bucket_name, key_prefix="ground-truth-data"):
    """Save JSON data directly to S3"""
    
    s3_client = boto3.client('s3')
    json_key = f"{key_prefix}/test_dataset.json"
    
    # Convert to JSON string if it's a list/dict
    if isinstance(data, (list, dict)):
        json_data = json.dumps(data, indent=2)
    else:
        json_data = data  # Already a string
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=json_key,
        Body=json_data,
        ContentType='application/json'
    )
    
    print(f"Saved dataset to S3: s3://{bucket_name}/{json_key}")
    return f"s3://{bucket_name}/{json_key}"

def get_sagemaker_default_bucket():
    """Get the default SageMaker bucket for the current session"""
    try:
        session = sagemaker.Session()
        bucket_name = session.default_bucket()
        print(f"Using SageMaker default bucket: {bucket_name}")
        return bucket_name
    except Exception as e:
        print(f"Error getting SageMaker default bucket: {e}")
        print("Falling back to manual bucket specification")
        return None



def upload_vector_db_to_s3(local_path, bucket_name, key_prefix="10k-vec-db"):
    """Upload the entire vector database directory to S3"""
    s3_client = boto3.client('s3')
    
    uploaded_files = []
    
    # Walk through all files in the vector db directory
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create relative path for S3 key
            relative_path = os.path.relpath(local_file_path, local_path)
            s3_key = f"{key_prefix}/{relative_path}"
            
            # Upload file
            s3_client.upload_file(local_file_path, bucket_name, s3_key)
            uploaded_files.append(f"s3://{bucket_name}/{s3_key}")
    
    print(f"Uploaded {len(uploaded_files)} files to S3 under s3://{bucket_name}/{key_prefix}/")
    return f"s3://{bucket_name}/{key_prefix}/"

def run():
    # Load the ground truth data
    data = load_json_data()
    
    if data is None:
        print("Failed to load ground truth data")
        return
    
    # Display sample
    print(f"\nSample data (first item):")
    print(json.dumps(data[0], indent=2))
    
    # Get SageMaker default bucket
    bucket_name = get_sagemaker_default_bucket()
    
    if bucket_name:
        # Upload test dataset
        dataset_s3_uri = save_to_s3(data, bucket_name)
        print(f"\nDataset saved to S3: {dataset_s3_uri}")
        
        # Upload existing vector database
        vec_db_path = os.path.join(os.path.dirname(__file__), '10k-vec-db')
        if os.path.exists(vec_db_path):
            vec_db_s3_uri = upload_vector_db_to_s3(vec_db_path, bucket_name)
            print(f"Vector database saved to S3: {vec_db_s3_uri}")
            
            return {
                "dataset_uri": dataset_s3_uri,
                "vector_db_uri": vec_db_s3_uri
            }
        else:
            print(f"Vector database not found at: {vec_db_path}")
            print("Please create the vector database first using your notebook")
            return {"dataset_uri": dataset_s3_uri}
    else:
        print("\nTo save to S3, specify your bucket name:")
        print("bucket_name = 'your-bucket-name'")

if __name__ == "__main__":
    run()