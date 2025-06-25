from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Verificar as dimensões dos dados   X-> imagens, Y-> rótulos
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

#Normalizar os valores dos pixels para o intervalo [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

#Codificar os rótulos com one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construindo a MLP
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

#Verificar acurácia do modelo no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acurácia no teste: {test_acc:.4f}")


predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Visualizar as predições das 10 primeiras imagens do teste

for i in range(10):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Verdadeiro: {true_classes[i]}, Previsto: {predicted_classes[i]}")
    plt.axis('off')
    plt.show()


# Matriz de Confusão
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# Distribuição de acertos e erros por classe
correct = np.where(predicted_classes == true_classes)[0]
incorrect = np.where(predicted_classes != true_classes)[0]

plt.figure(figsize=(12, 6))
plt.hist(true_classes[correct], bins=np.arange(11)-0.5, alpha=0.7, label='Corretos', color='g')
plt.hist(true_classes[incorrect], bins=np.arange(11)-0.5, alpha=0.7, label='Incorretos', color='r')
plt.xlabel('Classe')
plt.ylabel('Frequência')
plt.title('Distribuição de Acertos e Erros por Classe')
plt.legend()
plt.show()

# Exibição de imagens com erros de classificação
num_errors = len(incorrect)
plt.figure(figsize=(15, 15))
for i, idx in enumerate(incorrect[:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"V: {true_classes[idx]}, P: {predicted_classes[idx]}")
    plt.axis('off')
plt.suptitle('Erros de Classificação')
plt.show()

# Histograma das probabilidades de predição
plt.figure(figsize=(12, 6))
plt.hist(np.max(predictions, axis=1), bins=20, color='b', alpha=0.7)
plt.xlabel('Probabilidade de Predição')
plt.ylabel('Frequência')
plt.title('Histograma das Probabilidades de Predição')
plt.show()

# Visualização em 2D com PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_test.reshape(-1, 28*28))
plt.figure(figsize=(12, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=true_classes, cmap='tab10', alpha=0.7)
plt.colorbar()
plt.title('Visualização 2D com PCA')
plt.show()

# Visualização em 2D com t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40)
tsne_result = tsne.fit_transform(x_test.reshape(-1, 28*28))
plt.figure(figsize=(12, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=true_classes, cmap='tab10', alpha=0.7)
plt.colorbar()
plt.title('Visualização 2D com t-SNE')
plt.show()
