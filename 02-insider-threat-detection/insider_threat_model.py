import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("insider_threat_data.csv")

X = data[
    [
        "failed_logins",
        "after_hours_access",
        "large_downloads",
        "unusual_file_access",
        "policy_violations",
        "privileged_account",
        "prior_incidents",
    ]
]

y = data["threat_level"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(7,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=50, verbose=0)

new_activity = [[5, 1, 1, 1, 1, 1, 1]]
prediction = model.predict(new_activity)

predicted_class = prediction.argmax()
predicted_label = encoder.inverse_transform([predicted_class])

print("Predicted Insider Threat Level:", predicted_label[0])
