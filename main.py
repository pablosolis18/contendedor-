import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generar datos sintÃ©ticos
X, y = make_classification(n_samples=500, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelo entrenado con precisiÃ³n: {accuracy:.2f}")

# Guardar el modelo entrenado
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# FunciÃ³n para hacer predicciones
def predict(inputs):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([inputs])
    return prediction[0]

# Interfaz en la terminal
if __name__ == "__main__":
    print("\nIntroduce 5 valores separados por espacios para hacer una predicciÃ³n:")
    user_input = input().strip().split()
    user_input = np.array(user_input, dtype=float)
    result = predict(user_input)
    print(f"PredicciÃ³n del modelo: {result}")
