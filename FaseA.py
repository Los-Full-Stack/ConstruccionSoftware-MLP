import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. Implementación de un clasificador Perceptron simple
print("### Fase A: El Perceptron (Modelo de Capa Única) ###")
print("1. Implementación de un clasificador basado en un Perceptron simple (usando sklearn.linear_model.Perceptron)\n")

# Datos XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0]) # Salidas esperadas para XOR

print("Datos de entrada (X):\n", X)
print("Salidas esperadas (y):\n", y)
print("\n")

# Inicializar y entrenar el Perceptron
# random_state para reproducibilidad y tol=None para asegurar que se ejecute un número fijo de épocas si no converge
perceptron_model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
print("2. Intentando entrenar el modelo con los datos XOR mencionados...")
perceptron_model.fit(X, y)
print("Entrenamiento del Perceptron completado.\n")

# Realizar predicciones
y_pred = perceptron_model.predict(X)

# Evaluar el modelo
accuracy = accuracy_score(y, y_pred)

print("3. Observando el resultado:")
print("Predicciones del modelo para los datos XOR:\n", y_pred)
print(f"Precisión (Accuracy) del modelo: {accuracy * 100:.2f}%\n")

# Observaciones
print("### Análisis de los resultados ###")
print("¿Logra el modelo converger con una precisión del 100%? ¿Por qué?")

if accuracy < 1.0:
    print("El modelo NO logra converger con una precisión del 100% para los datos XOR.")
    print("\nEsto se debe a que la función XOR no es linealmente separable. Un Perceptron simple")
    print("es un clasificador lineal, lo que significa que solo puede encontrar una única línea (o hiperplano)")
    print("para separar las clases. Para los datos XOR, no es posible dibujar una sola línea recta")
    print("que separe los puntos (0,0) y (1,1) de los puntos (0,1) y (1,0) de manera perfecta.\n")
    print("Para resolver problemas no linealmente separables como XOR, se necesitan modelos más complejos")
    print("como las redes neuronales multicapa (MLP) con funciones de activación no lineales.")
else:
    print("¡Sorprendentemente, el modelo ha alcanzado el 100% de precisión!")
    print("Esto es inusual para XOR con un Perceptron simple y podría indicar alguna particularidad en la inicialización o en los parámetros de entrenamiento.")
    print("Normalmente, un Perceptron de una sola capa no puede clasificar el problema XOR con 100% de precisión debido a su naturaleza no linealmente separable.")