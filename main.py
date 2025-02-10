from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Entrenar y guardar el modelo al iniciar el contenedor
X, y = make_classification(n_samples=500, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Guardar el modelo entrenado
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Cargar el modelo una sola vez en memoria
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Â¡Bienvenido a la API de predicciÃ³n con Flask!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Usar el modelo cargado en memoria
        prediction = model.predict(features)

        return jsonify({"prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
