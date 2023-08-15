from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  # Import the SimpleImputer class

app = Flask(__name__)

# Load the data
df = pd.read_csv("D:\\iris.csv")  # Update the path to your dataset
df_2 = pd.read_csv("D:\\delaney_solubility_with_descriptors.csv")
df_3 = pd.read_csv("D:\\NBA.csv")

# Drop the "Id" column
df = df.drop("Id", axis=1)
df_3 = df_3.drop("Tm", axis=1)
df_3 = df_3.drop("Pos", axis=1)

# Split data into features (X) and target (y)
X = df.drop("Species", axis=1)
y = df["Species"]
x_2 = df_2.drop("logS", axis=1)
y_2 = df_2["logS"]
x_3 = df_3.drop("Player", axis=1)
y_3 = df_3["Player"]

# Create an imputer with a chosen strategy (e.g., mean or median)
imputer = SimpleImputer(strategy="mean")  # You can also use "median", "most_frequent", etc.

# Apply the imputer to your feature data
x_3_imputed = imputer.fit_transform(x_3)

# Encode the categorical labels into numeric values
label_encoder_species = LabelEncoder()
label_encoder_solubility = LabelEncoder()
label_encoder_NBA = LabelEncoder()

y_encoded = label_encoder_species.fit_transform(y)
y_2_encoded = label_encoder_solubility.fit_transform(y_2)
y_3_encoded = label_encoder_NBA.fit_transform(y_3)

# Create and fit a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X, y_encoded)
rf2_classifier = RandomForestClassifier(random_state=42)
rf2_classifier.fit(x_2, y_2_encoded)
rf3_classifier = RandomForestClassifier(random_state=42)
rf3_classifier.fit(x_3_imputed, y_3_encoded)  # Use the imputed data

@app.route("/data.html", methods=["GET", "POST"])
def data():
    if request.method == "POST":
        MolLogP = float(request.form["MolLogP"])
        MolW = float(request.form["MolW"])
        NumRotatable = float(request.form["NumRotatable"])
        Aromatic = float(request.form["Aromatic"])
        
        predicted_solubility_result  = predict_solubility(MolLogP, MolW, NumRotatable, Aromatic)
        
        return render_template("data.html", predict_solubility=predicted_solubility_result )
    
    return render_template('data.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/NBA.html", methods=["GET", "POST"])
def NBA():
    if request.method == "POST":
        age = float(request.form["age"])
        G = float(request.form["G"])
        GS = float(request.form["GS"])
        MP = float(request.form["MP"])
        FG = float(request.form["FG"])
        FGA = float(request.form["FGA"])
        FG_percent = float(request.form["FG_percent"])
        ThreeP = float(request.form["ThreeP"])
        ThreePA = float(request.form["ThreePA"])
        ThreeP_percent = float(request.form["ThreeP_percent"])
        TwoP = float(request.form["TwoP"])
        TwoPA = float(request.form["TwoPA"])
        TwoP_percent = float(request.form["TwoP_percent"])
        eFG_percent = float(request.form["eFG_percent"])
        FT = float(request.form["FT"])
        FTA = float(request.form["FTA"])
        FT_percent = float(request.form["FT_percent"])
        ORB = float(request.form["ORB"])
        DRB = float(request.form["DRB"])
        TRB = float(request.form["TRB"])
        AST = float(request.form["AST"])
        STL = float(request.form["STL"])
        BLK = float(request.form["BLK"])
        TOV = float(request.form["TOV"])
        PF = float(request.form["PF"])
        PTS = float(request.form["PTS"])
        
        predict_NBA_results = predict_NBA(age, G, GS, MP, FG, FGA, FG_percent, ThreeP, ThreePA, ThreeP_percent, TwoP, TwoPA, TwoP_percent, eFG_percent, FT, FTA, FT_percent, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS)
        
        return render_template("NBA.html", predict_NBA=predict_NBA_results)
    
    return render_template("NBA.html")


@app.route("/iris.html", methods=["GET", "POST"])
def iris():
    if request.method == "POST":
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
        
        predicted_species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        
        return render_template("iris.html", predicted_species=predicted_species)
    
    return render_template("iris.html")

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    predicted_class = rf_classifier.predict(input_features)
    predicted_species = label_encoder_species.inverse_transform(predicted_class)
    return predicted_species[0]

def predict_solubility(MolLogP, MolW, NumRotatable, Aromatic):
    input_features = [[MolLogP, MolW, NumRotatable, Aromatic]]
    predicted_class = rf2_classifier.predict(input_features)
    predicted_solubility = label_encoder_solubility.inverse_transform(predicted_class)
    return predicted_solubility[0]

def predict_NBA(age, G, GS, MP, FG, FGA, FG_percent, ThreeP, ThreePA, ThreeP_percent, TwoP, TwoPA, TwoP_percent, eFG_percent, FT, FTA, FT_percent, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS):
    input_features = [[age, G, GS, MP, FG, FGA, FG_percent, ThreeP, ThreePA, ThreeP_percent, TwoP, TwoPA, TwoP_percent, eFG_percent, FT, FTA, FT_percent, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS]]
    predicted_class = rf3_classifier.predict(input_features)
    predicted_NBA = label_encoder_NBA.inverse_transform(predicted_class)
    return predicted_NBA[0]



if __name__ == "__main__":
    app.run(debug=True,port=8000)
