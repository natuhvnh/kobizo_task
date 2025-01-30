import pandas as pd
import os

#
def append_model_output(new_df, file_path):
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    # Write the updated DataFrame back to the Parquet file
    updated_df.to_parquet(file_path, index=False)
    print("WRITE MODEL OUTPUT = DONE")
    return
