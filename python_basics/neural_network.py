import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """SIMPLE 2 LAYER NEURAL NETWORK"""

    def __init__(self,input_size,hidden_size,output_size,learning_rate=0.1):
        """"INITIALIZE NETWORK"""

        self.W1=np.random.randn(input_size,hidden_size)*0.5
        self.b1=np.zeros((1,hidden_size))

        self.W2=np.random.randn(hidden_size,output_size)
        self.b2=np.zeros((1,output_size))

        self.learning_rate=learning_rate

        self.losses=[]
        pass
    def sigmoid(self,x):
        """SIGMOND ACTIVATION FUNCTION"""
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,X):
        """DERIVATIVE OF SIGMOID"""
        return X*(1-X)
    
    def forward(self,X):
        """FORWARD PROPOGATION"""
        self.z1=np.dot(X,self.W1)+self.b1
        self.a1=self.sigmoid(self.z1)

        self.z2=np.dot(self.a1,self.W2)+self.b2
        self.a2=self.sigmoid(self.z2)

        return self.a2
    
    def backward(self,X,y,output):
        """BACKWARD PROPAGATION"""

        m=X.shape[0]

        output_error=output-y
        output_delta=output_error*self.sigmoid_derivative(output)
        hidden_error=np.dot(output_delta,self.W2.T)
        hidden_delta=hidden_error*self.sigmoid_derivative(self.a1)


        self.W2-=self.learning_rate*np.dot(self.a1.T,output_delta)/m
        self.b2-=self.learning_rate*np.sum(output_delta,axis=0,keepdims=True)/m

        self.W1-=self.learning_rate*np.dot(X.T,hidden_delta)/m
        self.b1-=self.learning_rate*np.sum(hidden_delta,axis=0,keepdims=True)/m
    
    def train(self,X,y,epochs=10000,verbose=True):
        """TRAIN THE NETWORk"""

        for epoch in range(epochs):
            output=self.forward(X)

            loss=np.mean((output-y)**2)
            self.losses.append(loss)

            self.backward(X,y,output)

            if verbose and (epoch%1000==0 or epoch==epochs-1):
                print(f"Epoch {epoch:5d} | Loss:{loss:.6f}")

    def predict(self,X):
        """MAKE PREDICTIONS"""
        output=self.forward(X)
        return (output>0.5).astype(int)
    
    def plot_loss(self):
        """PLOT TRAINING LOSS"""
        plt.figure(figsize=(10,5))
        plt.plot(self.losses)
        plt.title("TRAINING LOSS OVER TIME")
        plt.xlabel("EPOCH")
        plt.ylabel("LOSS (MSE)")
        plt.grid(True)
        plt.show()

print("="*50)
print("NEURAL NETWORK FROM SCRATCH")
print("Solving XOR Problems")
print("="*50)

#TRAINING DATA
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("\nTRAINING DATA (XOR:")
print("INPUT | OUTPUT")
print("------|-------")

for i in range(len(X)):
    print(f"{X[i][0]} {X[i][1]}  |   {y[i][0]}")

print("\n"+"="*50)
print("TRAINING. . .")
print("="*50 + "\n")

#CREATING AND TRAINING
nn=NeuralNetwork(input_size=2,hidden_size=4,output_size=1,learning_rate=0.5)
nn.train(X,y,epochs=10000,verbose=True)

#NETWORK TESTING
print("\n"+"="*50)
print("PREDICTIONS")
print("="*50)
print("\nInput | Target | Prediction | Probability")
print("------|--------|------------|------------")

predictions=nn.predict(X)
probabilities=nn.forward(X)

for i in range(len(X)):
    print(f"{X[i][0]} {X[i][1]}   |   {y[i][0]}    |     {predictions[i][0]}      |   {probabilities[i][0]:.4f}")

accuracy=np.mean(predictions==y)*100
print(f"\nAccuracy: {accuracy:.1f}%")

nn.plot_loss()

def plot_decision_boundary(nn,X,y):

    x_min,x_max=-0.5,1.5
    y_min,y_max=-0.5,1.5
    h=0.01

    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    Z=nn.forward(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)

    plt.figure(figsize=(10,8))
    plt.contourf(xx,yy,Z,levels=20,cmap='RdYlBu',alpha=0.8)
    plt.colorbar(label='Output Probability')

    #plotting training points
    scatter=plt.scatter(X[:, 0], X[:, 1], c=y, s=200, edgecolors='black', linewidths=2,cmap='RdYlBu', vmin=0, vmax=1)

    plt.xlabel('Input 1', fontsize=12)
    plt.ylabel('Input 2', fontsize=12)
    plt.title('Neural Network Decision Boundary (XOR Problem)', fontsize=14)
    plt.grid(True, alpha=0.3)

    for i in range(len(X)):
         plt.annotate(f"({X[i][0]}, {X[i][1]}) → {y[i][0]}", 
                    xy=(X[i][0], X[i][1]), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    plt.show()

print("\nPlotting decision boundary...")
plot_decision_boundary(nn, X, y)

print("\n" + "="*50)
print("YOU JUST BUILT A NEURAL NETWORK FROM SCRATCH!")
print("="*50)


    