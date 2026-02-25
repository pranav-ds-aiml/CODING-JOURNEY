import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate
        self.losses = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]

        # BCE + Sigmoid simplifies to this
        output_delta = output - y

        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        self.W2 -= self.learning_rate * np.dot(self.a1.T, output_delta) / m
        self.b2 -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * np.dot(X.T, hidden_delta) / m
        self.b1 -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)

            # Binary Cross Entropy Loss
            epsilon = 1e-8
            loss = -np.mean(
                y * np.log(output + epsilon) +
                (1 - y) * np.log(1 - output + epsilon)
            )
            self.losses.append(loss)

            self.backward(X, y, output)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Training Loss (BCE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()


# XOR Dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Train
nn = NeuralNetwork(2, 4, 1)
nn.train(X, y)

# Test
predictions = nn.predict(X)
probabilities = nn.forward(X)

print("\nInput | Target | Prediction | Probability")
print("------------------------------------------")
for i in range(len(X)):
    print(f"{X[i]}   |   {y[i][0]}   |     {predictions[i][0]}      |   {probabilities[i][0]:.4f}")

accuracy = np.mean(predictions == y) * 100
print(f"\nAccuracy: {accuracy:.1f}%")

nn.plot_loss()