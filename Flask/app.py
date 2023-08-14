from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the data
df = pd.read_csv("D:\iris.csv")  # Update the path to your dataset

# Drop the "Id" column
df = df.drop("Id", axis=1)

# Split data into features (X) and target (y)
X = df.drop("Species", axis=1)
y = df["Species"]

# Encode the categorical labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create and fit a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X, y_encoded)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
        
        predicted_species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        
        return render_template("index.html", predicted_species=predicted_species)
    
    return render_template("index.html")

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    predicted_class = rf_classifier.predict(input_features)
    predicted_species = label_encoder.inverse_transform(predicted_class)
    return predicted_species[0]

if __name__ == "__main__":
    app.run(debug=True,port=8000)
