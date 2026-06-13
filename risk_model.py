import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Read the dataset
data = pd.read_csv("security_events.csv")

# Features
X = data[
    [
        "failed_logins",
        "after_hours_access",
        "unusual_file_access",
        "policy_violations",
        "privileged_account"
    ]
]

# Labels
y = data["risk_level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

# Example investigation
new_event = [[4, 1, 1, 1, 1]]

prediction = model.predict(new_event)

print("Predicted Risk Level:", prediction[0])
