from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

# ------------------------
# Load & preprocess dataset
# ------------------------
data = [
    ["AR/VR/Gaming", "LTE/5G", "LTE/5G", "GBR", 0.001, 50, "eMBB"],
    ["AR/VR/Gaming", "LTE/5G", "LTE/5G", "Non-GBR", 0.001, 100, "eMBB"],
    ["Healthcare", "IoT", "IoT", "GBR", 0.000001, 10, "URLLC"],
    ["Industry 4.0", "IoT", "IoT", "Non-GBR", 0.001, 50, "mMTC"],
    ["Industry 4.0", "IoT", "IoT", "GBR", 0.000001, 10, "URLLC"],
    ["IoT Devices", "IoT", "IoT", "GBR", 0.01, 50, "mMTC"],
    ["IoT Devices", "IoT", "IoT", "Non-GBR", 0.01, 300, "mMTC"],
    ["Public Safety/E911", "LTE/5G", "LTE/5G", "GBR", 0.000001, 10, "URLLC"],
    ["Smart City & Home", "IoT", "IoT", "GBR", 0.01, 50, "mMTC"],
    ["Smart City & Home", "IoT", "IoT", "Non-GBR", 0.01, 300, "mMTC"],
    ["Smart Transportation", "IoT", "IoT", "GBR", 0.000001, 10, "URLLC"],
    ["Smartphone", "LTE/5G", "LTE/5G", "GBR", 0.01, 75, "eMBB"],
    ["Smartphone", "LTE/5G", "LTE/5G", "GBR", 0.01, 100, "eMBB"],
    ["Smartphone", "LTE/5G", "LTE/5G", "GBR", 0.001, 150, "eMBB"],
    ["Smartphone", "LTE/5G", "LTE/5G", "GBR", 0.000001, 300, "eMBB"],
    ["Smartphone", "LTE/5G", "LTE/5G", "Non-GBR", 0.000001, 60, "eMBB"],
    ["Smartphone", "LTE/5G", "LTE/5G", "Non-GBR", 0.000001, 100, "eMBB"],
    ["Smartphone", "LTE/5G", "LTE/5G", "Non-GBR", 0.000001, 300, "eMBB"]
]
columns = ["UseCase", "UE_Category", "Tech_Supported", "GBR", "PacketLoss", "Delay", "SliceType"]
df = pd.DataFrame(data, columns=columns)

label_encoders = {}
for col in ["UseCase", "UE_Category", "Tech_Supported", "GBR", "SliceType"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("SliceType", axis=1)
y = df["SliceType"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

@app.route("/", methods=["GET", "POST"])
def index():
    results = {}
    prediction = None
    best_model_name = None
    category_detected = None

    if request.method == "POST":
        # Get user input
        use_case = request.form["use_case"]
        ue_category = request.form["ue_category"]
        tech_supported = request.form["tech_supported"]
        gbr = request.form["gbr"]
        packet_loss = float(request.form["packet_loss"])
        delay = float(request.form["delay"])

        # Determine category based on delay and loss
        if delay < 20 and packet_loss < 0.001:
            category_detected = "URLLC"
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Naive Bayes": GaussianNB(),
                "Decision Tree (shallow)": DecisionTreeClassifier(max_depth=3)
            }
        elif delay >= 20 and delay <= 200:
            category_detected = "eMBB"
            models = {
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                "Decision Tree": DecisionTreeClassifier()
            }
        else:
            category_detected = "mMTC"
            models = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Bagging Trees": BaggingClassifier(),
                "MLP": MLPClassifier(max_iter=1000)
            }

        # Evaluate models
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, preds)
            results[name] = round(acc, 4)

        # Select best model
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]

        # Prepare new packet for prediction
        new_packet = pd.DataFrame([[use_case, ue_category, tech_supported, gbr, packet_loss, delay]],
                                  columns=["UseCase", "UE_Category", "Tech_Supported", "GBR", "PacketLoss", "Delay"])

        for col in ["UseCase", "UE_Category", "Tech_Supported", "GBR"]:
            new_packet[col] = label_encoders[col].transform(new_packet[col])

        new_packet_scaled = scaler.transform(new_packet)
        pred_label_encoded = best_model.predict(new_packet_scaled)[0]
        prediction = label_encoders["SliceType"].inverse_transform([pred_label_encoded])[0]

    return render_template("index.html",
                           results=results,
                           prediction=prediction,
                           best_model=best_model_name,
                           category=category_detected)

if __name__ == "__main__":
    app.run(debug=True)
