import pandas as pd
import re
import os

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# Drop rows with missing Msg Type
df_msg = df.dropna(subset=["Msg Type"]).copy()

# Clean Msg Type by removing namespaces/paths, keeping only final type name
def clean_msg_types(msg):
    parts = [re.sub(r".*[:\.]", "", m.strip()) for m in msg.split(" and ")]
    return " and ".join(parts)

# Apply cleaning function
df_msg["Msg Type Clean"] = df_msg["Msg Type"].apply(clean_msg_types)

# Define alternative path
output_dir = "data/"
output_filename = "qualitative_clean.csv"
output_path = os.path.join(output_dir, output_filename)

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save full CSV with cleaned message types
df_msg.to_csv(output_path, index=False)

print(f"Cleaned CSV saved to: {output_path}")
