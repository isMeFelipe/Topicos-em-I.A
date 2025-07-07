# Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
    X, y_encoded, test_size=0.2, random_state=42,
)

# Definir os modelos
models = {
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'K-Vizinhos mais Próximos (KNN)': KNeighborsClassifier(),
    'Floresta Aleatória (RF)': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Máquina de Vetores de Suporte (SVM)': SVC(random_state=42)
}

# Comparação de Desempenho entre os modelos
for model_name, model in models.items():
    print(f"\nTreinando modelo: {model_name}")
    
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Previsões no conjunto de teste
    y_pred = model.predict(X_test)
    
    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.3f}")
    
    # Métricas detalhadas: Precisão, Revocação, F1
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(cm)
    
    print("\nDetalhes sobre acertos e erros por classe:")
    for i in range(len(cm)):
        print(f"\nClasse {i}:")
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive
        true_negative = cm.sum() - (true_positive + false_positive + false_negative)
        
        accuracy_class = true_positive / (true_positive + false_positive + false_negative)
        print(f"  Acurácia da classe {i}: {accuracy_class:.3f}")
        
        error_class = (false_positive + false_negative) / (true_positive + false_positive + false_negative)
        print(f"  Taxa de erro da classe {i}: {error_class:.3f}")
    
    # Relatório de Classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

# Ajuste de Parâmetros com GridSearchCV

# Parâmetros para ajuste
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

# Modelos a serem ajustados
models_to_tune = {
    'K-Vizinhos mais Próximos (KNN)': KNeighborsClassifier(),
    'Floresta Aleatória (RF)': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Máquina de Vetores de Suporte (SVM)': SVC(random_state=42)
}

# Ajuste de Parâmetros com GridSearchCV
best_models = {}

for model_name, model in models_to_tune.items():
    print(f"\nAjustando parâmetros para {model_name}")
    
    if model_name == 'K-Vizinhos mais Próximos (KNN)':
        grid_search = GridSearchCV(model, param_grid_knn, cv=5, n_jobs=-1, verbose=1)
    elif model_name == 'Floresta Aleatória (RF)':
        grid_search = GridSearchCV(model, param_grid_rf, cv=5, n_jobs=-1, verbose=1)
    elif model_name == 'Máquina de Vetores de Suporte (SVM)':
        grid_search = GridSearchCV(model, param_grid_svm, cv=5, n_jobs=-1, verbose=1)
    
    # Treinar com busca em grade
    grid_search.fit(X_train, y_train)
    
    # Melhor modelo encontrado
    best_models[model_name] = grid_search.best_estimator_
    print(f"Melhor modelo para {model_name}: {grid_search.best_params_}")

# Avaliar os melhores modelos ajustados
for model_name, best_model in best_models.items():
    print(f"\nModelo ajustado: {model_name}")
    
    # Previsões no conjunto de teste
    y_pred = best_model.predict(X_test)
    
    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.3f}")
    
    # Métricas detalhadas: Precisão, Revocação, F1
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(cm)
    
    print("\nDetalhes sobre acertos e erros por classe:")
    for i in range(len(cm)):
        print(f"\nClasse {i}:")
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive
        true_negative = cm.sum() - (true_positive + false_positive + false_negative)
        
        accuracy_class = true_positive / (true_positive + false_positive + false_negative)
        print(f"  Acurácia da classe {i}: {accuracy_class:.3f}")
        
        error_class = (false_positive + false_negative) / (true_positive + false_positive + false_negative)
        print(f"  Taxa de erro da classe {i}: {error_class:.3f}")
    
    # Relatório de Classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

# Importância dos atributos para a Árvore de Decisão
print("\nImportância dos Atributos para a Árvore de Decisão:")
if 'Árvore de Decisão' in models:
    tree_model = models['Árvore de Decisão']
    importances = tree_model.feature_importances_
    features = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print(features)

# Visualização da árvore de decisão com dtreeviz (apenas para Árvore de Decisão)
if 'Árvore de Decisão' in models:
    viz_model = dtreeviz_model(
        tree_model,
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
sample_preds = best_models['K-Vizinhos mais Próximos (KNN)'].predict(sample_instances)

print("\nPredições para 3 instâncias aleatórias do conjunto de teste:")
for idx, instance_idx in enumerate(sample_idxs):
    instance = X_test.loc[instance_idx]
    predicted_class = label_encoder.inverse_transform([sample_preds[idx]])[0]
    print(f"\nInstância {idx + 1} (índice original: {instance_idx}):")
    print(instance)
    print("Classe predita:", predicted_class)
