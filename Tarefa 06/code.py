# Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder

from dtreeviz import model as dtreeviz_model

# Carregar dataset com separador correto
df = pd.read_csv("dataset.csv", sep=";")
df = df.sample(n=1000, random_state=42)  # Reduz amostra para facilitar visualização

# Renomear a coluna da classe
df.rename(columns={'quality': 'classe'}, inplace=True)

# Separar preditores e alvo
X = df.drop(columns=['classe'])
y = df['classe']

# Codificar a classe como valores contínuos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir dados (80% treino / 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42,
)

# Treinar modelo de Árvore de Decisão
model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Matriz de Confusão:")
print(cm)
print("\nDetalhes sobre acertos e erros por classe:")

# Para cada classe
for i in range(len(cm)):
    print(f"\nClasse {i}:")
    
    # Verdadeiros positivos (acertos)
    true_positive = cm[i, i]
    # Falsos positivos (predições erradas para a classe i)
    false_positive = cm[:, i].sum() - true_positive
    # Falsos negativos (não previu a classe i quando deveria)
    false_negative = cm[i, :].sum() - true_positive
    # Verdadeiros negativos (acertos nas outras classes)
    true_negative = cm.sum() - (true_positive + false_positive + false_negative)

    # Exibição dos valores
    print(f"  Verdadeiros positivos (VP): {true_positive}")
    print(f"  Falsos positivos (FP): {false_positive}")
    print(f"  Falsos negativos (FN): {false_negative}")
    print(f"  Verdadeiros negativos (VN): {true_negative}")
    
    # Taxa de acerto para a classe i
    accuracy_class = true_positive / (true_positive + false_positive + false_negative)
    print(f"  Acurácia da classe {i}: {accuracy_class:.3f}")
    
    # Taxa de erro para a classe i
    error_class = (false_positive + false_negative) / (true_positive + false_positive + false_negative)
    print(f"  Taxa de erro da classe {i}: {error_class:.3f}")

# Exibindo a matriz de confusão
print("\nMatriz de Confusão Completa:")
print(cm)

print("\nRelatório de Classificação:")

# Filtra as classes realmente presentes no y_test
present_classes = np.unique(y_test)
class_names_present = label_encoder.inverse_transform(present_classes)

print(classification_report(
    y_test,
    y_pred,
    labels=present_classes,
    target_names=class_names_present.astype(str),
    digits=3
))


# Métricas específicas para a classe minoritária
unique, counts = np.unique(y_test, return_counts=True)
min_class = unique[np.argmin(counts)]

# Importância dos atributos
importances = model.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nImportância dos Atributos:")
print(features)

# Visualização da árvore com dtreeviz
viz_model = dtreeviz_model(
    model,
    X_train=X_train,
    y_train=y_train,
    feature_names=list(X.columns),
    target_name="classe",
    class_names=[str(c) for c in label_encoder.classes_]
)


v = viz_model.view()
v.show()
v.save("arvore_dtreeviz.svg")

# Predizer 3 instâncias aleatórias do conjunto de teste
sample_idxs = np.random.choice(X_test.index, 3, replace=False)
sample_instances = X_test.loc[sample_idxs]
sample_preds = model.predict(sample_instances)

print("\nPredições para 3 instâncias aleatórias do conjunto de teste:")
for idx, instance_idx in enumerate(sample_idxs):
    instance = X_test.loc[instance_idx]
    predicted_class = label_encoder.inverse_transform([sample_preds[idx]])[0]
    print(f"\nInstância {idx + 1} (índice original: {instance_idx}):")
    print(instance)
    print("Classe predita:", predicted_class)
