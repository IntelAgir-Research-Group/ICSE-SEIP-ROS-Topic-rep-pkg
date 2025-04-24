import pandas as pd

# Load the CSV file
file_path = "run_table-v1.csv"
df = pd.read_csv(file_path)

# Filter the DataFrame to keep only the specified msg_type values
filtered_df = df[df["msg_type"].isin(["PointCloud", "PointCloud2", "LaserScan", "Float64MultiArray"])]

filtered_df.to_csv("filtered_run_table.csv", index=False)
