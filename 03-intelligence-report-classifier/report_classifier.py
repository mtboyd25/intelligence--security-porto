import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the intelligence reports
data = pd.read_csv("intelligence_reports.csv")

# Display the first few reports
print("=== Intelligence Reports ===")
print(data.head())

# Encode intelligence categories
encoder = LabelEncoder()

data["encoded_type"] = encoder.fit_transform(
    data["intelligence_type"]
)

print("\n=== Intelligence Categories ===")
print(data[["intelligence_type", "encoded_type"]])

print("\nCategories Learned:")
print(encoder.classes_)
