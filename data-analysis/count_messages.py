import pandas as pd
import re

file_path = "data/qualitative_analysis.csv"
df = pd.read_csv(file_path)

df_msg = df.dropna(subset=["Msg Type"]).copy()

def clean_msg_types(msg):
    return [re.sub(r".*[:\.]", "", m.strip()) for m in msg.split(" and ")]

df_msg["Msg Type"] = df_msg["Msg Type"].apply(clean_msg_types)
df_exploded = df_msg.explode("Msg Type")

msg_type_counts = df_exploded["Msg Type"].value_counts().reset_index()
msg_type_counts.columns = ["Message Type", "Count"]

total_unique = msg_type_counts.shape[0]

print(f"Total unique message types: {total_unique}\n")
print("Top message types by frequency:\n")
print(msg_type_counts.head(30))  # Top 20, change as needed

msg_type_counts.to_csv("data/message_type_rankings.csv", index=False)
