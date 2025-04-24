import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# Drop rows with missing 'Field' values
df_field = df.dropna(subset=["Field"]).copy()

# Split multiple fields (e.g., "Education and Simulation") and explode into rows
df_field["Field"] = df_field["Field"].str.split(" and ")
df_field_exploded = df_field.explode("Field")

# Count the number of occurrences per field
field_counts = df_field_exploded["Field"].value_counts().reset_index()
field_counts.columns = ["Field", "Count"]

# Plot the barplot
plt.figure(figsize=(5, 3))
sns.barplot(data=field_counts, x="Field", y="Count", palette="Greens_r")

# Formatting
#plt.title("Distribution of Fields of Application")
plt.xlabel("")
plt.ylabel("Number of Projects")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

# Save to PDF
output_path = "graphs/qualitative/barplot_fields_of_application.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
