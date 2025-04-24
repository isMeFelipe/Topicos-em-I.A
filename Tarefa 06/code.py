# Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from dtreeviz import model as dtreeviz_model

# Carregar dataset com separador correto
df = pd.read_csv("dataset.csv", sep=";")
df = df.sample(n=1000, random_state=42)

# Renomear a coluna da classe
df.rename(columns={'quality': 'classe'}, inplace=True)

# Separar preditores e alvo
X = df.drop(columns=['classe'])
y = df['classe']

# Codificar a classe como valores contínuos começando de 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar modelo de árvore de decisão
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, digits=3))

# Visualização com dtreeviz (zoom nativo no navegador)
viz_model = dtreeviz_model(
    model,
    X_train=X_train,
    y_train=y_train,
    feature_names=list(X.columns),
    target_name="classe",
    class_names=[str(c) for c in label_encoder.classes_]
)

v = viz_model.view()
v.show()  # Exibe a visualização
v.save("arvore_dtreeviz.svg")  # Salva como SVG opcionalmente
