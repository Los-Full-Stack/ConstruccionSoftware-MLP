from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

print("\n### Fase B: El Perceptron Multicapa (MLP) ###")
print("1. Implementación de un clasificador MLP (usando sklearn.neural_network.MLPClassifier)\n")

# Datos XOR (re-declarados para asegurar disponibilidad, aunque ya estén en el kernel)
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

# 2. Configuración de la arquitectura del MLP
# Dos capas ocultas: la primera con 3 neuronas y la segunda con 2 neuronas, y función de activación ReLU
mlp_model = MLPClassifier(hidden_layer_sizes=(3, 2), activation='relu', max_iter=10000, random_state=100, solver='lbfgs') # Changed random_state

print("Configuración del MLP:")
print("- Capas ocultas: 2 capas: la primera con 3 neuronas y la segunda con 2 neuronas")
print("- Función de activación: ReLU (no lineal)\n")

print("3. Entrenando el modelo MLP con los datos XOR...")
mlp_model.fit(X, y)
print("Entrenamiento del MLP completado.\n")

# Realizar predicciones con el MLP
y_pred_mlp = mlp_model.predict(X)

# Evaluar el modelo MLP
accuracy_mlp = accuracy_score(y, y_pred_mlp)

print("Resultados del MLP:")
print("Predicciones del modelo MLP para los datos XOR:\n", y_pred_mlp)
print(f"Precisión (Accuracy) del modelo MLP: {accuracy_mlp * 100:.2f}%\n")

print("### Comparación y Análisis de Resultados del MLP ###")
if accuracy_mlp == 1.0:
    print("El modelo MLP ha logrado converger con una precisión del 100% para los datos XOR.")
    print("Esto demuestra la capacidad de las redes neuronales multicapa con funciones de activación no lineales")
    print("para resolver problemas no linealmente separables como el XOR.")
    print("Las capas ocultas y las funciones de activación no lineales permiten al MLP aprender")
    print("representaciones complejas y crear fronteras de decisión no lineales que un Perceptrón simple no puede.")
else:
    print("El modelo MLP no ha alcanzado el 100% de precisión. Esto podría deberse a la inicialización, la complejidad del modelo o la cantidad de iteraciones.")
    print("Sin embargo, es común que un MLP resuelva el XOR con alta precisión si está bien configurado.")
print("\n--- Prueba con datos personalizados --- ")

try:
    # Solicitar al usuario que ingrese dos valores
    input1 = int(input("Ingrese el primer valor (0 o 1): "))
    input2 = int(input("Ingrese el segundo valor (0 o 1): "))

    # Validar la entrada
    if input1 not in [0, 1] or input2 not in [0, 1]:
        print("Entrada inválida. Por favor, ingrese solo 0 o 1.")
    else:
        # Preparar los datos de entrada para la predicción
        custom_input = np.array([[input1, input2]])

        # Realizar la predicción con el modelo MLP entrenado
        # Asegurarse de que 'mlp_model' esté disponible en el entorno del kernel
        if 'mlp_model' in globals():
            custom_prediction = mlp_model.predict(custom_input)
            print(f"Para la entrada ({input1}, {input2}), la predicción del MLP es: {custom_prediction[0]}")

            # Calcular la salida XOR real para comparación
            real_xor_output = 1 if (input1 != input2) else 0
            print(f"La salida XOR real para ({input1}, {input2}) es: {real_xor_output}")

            if custom_prediction[0] == real_xor_output:
                print("¡El MLP predijo correctamente!")
            else:
                print("El MLP no predijo correctamente en este caso.")
        else:
            print("Error: El modelo MLP no ha sido entrenado o no está disponible. Por favor, ejecute la fase B primero.")

except ValueError:
    print("Entrada inválida. Por favor, asegúrese de ingresar números enteros.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")