import pandas as pd
import os
import math

def split_parquet(file_path, chunk_size_mb=50):
    # Read the file
    print(f"Reading {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Estimate size in memory might be different, but we care about file size.
    # A simple row-based split is usually sufficient.
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    num_chunks = math.ceil(file_size_mb / chunk_size_mb)
    
    print(f"File size: {file_size_mb:.2f} MB. Splitting into {num_chunks} chunks...")
    
    rows_per_chunk = math.ceil(len(df) / num_chunks)
    
    base_name = os.path.splitext(file_path)[0]
    
    for i in range(num_chunks):
        start_idx = i * rows_per_chunk
        end_idx = min((i + 1) * rows_per_chunk, len(df))
        
        chunk_df = df.iloc[start_idx:end_idx]
        chunk_name = f"{base_name}_part_{i}.parquet"
        
        print(f"Writing {chunk_name} ({len(chunk_df)} rows)...")
        chunk_df.to_parquet(chunk_name)
        
    print("Done!")

if __name__ == "__main__":
    split_parquet("tala_final.parquet")
