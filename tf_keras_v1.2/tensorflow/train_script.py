import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import numpy as np

# Przykładowe dane (X_train i y_train)
# Zastąp to rzeczywistymi danymi
X_train = np.random.rand(100, 30)  # 100 próbek, 30 cech
y_train = np.random.randint(0, 2, size=(100,))  # Binarna klasyfikacja (0 lub 1)

# Standardyzacja danych wejściowych
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Budowanie modelu
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],), 
          kernel_regularizer=l2(0.001)),  # Warstwa ukryta z regularyzacją L2
    Dense(1, activation='sigmoid')  # Warstwa wyjściowa dla klasyfikacji binarnej
])

# Kompilacja modelu
model.compile(optimizer='adam',  # Używamy optymalizatora Adam
              loss='binary_crossentropy',  # Funkcja straty entropii krzyżowej
              metrics=['accuracy'])  # Śledzimy dokładność jako metrykę

# Trenowanie modelu
history = model.fit(X_train_scaled, y_train, epochs=50000, validation_split=0.2)

# Możesz chcieć dodać callbacki, np. EarlyStopping, do wczesnego zatrzymywania
