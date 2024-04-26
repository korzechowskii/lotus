# Wybór obrazu bazowego
FROM python:3.8-slim

# Ustawienie katalogu roboczego w kontenerze
WORKDIR /app

# Instalacja TensorFlow oraz innych potrzebnych bibliotek
RUN pip install tensorflow scikit-learn

# Kopiowanie skryptu do kontenera
COPY train_script.py /app/

# Uruchomienie TensorBoard na porcie 6006
EXPOSE 6006

# Uruchomienie skryptu treningowego
CMD ["python", "train_script.py"]
