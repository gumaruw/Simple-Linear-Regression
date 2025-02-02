import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

def load_data(file_path):
    # Load data from a CSV file
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def generate_data(num_samples=100):
    np.random.seed(0)
    X = 2 * np.random.rand(num_samples, 1)
    y = 4 + 3 * X + np.random.randn(num_samples, 1)
    return X, y

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        gradients = X.T.dot(X.dot(theta) - y) / m
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

def main():
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    # Load or generate data
    use_generated_data = True
    if use_generated_data:
        X, y = generate_data()
    else:
        file_path = "data/dataset.csv"
        X, y = load_data(file_path)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add x0 = 1 to each instance
    X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
    
    # Initialize theta
    theta = np.random.randn(2, 1)
    
    # Perform gradient descent
    theta, cost_history = gradient_descent(X_train_b, y_train, theta)
    
    # Predict on the test set
    y_pred = X_test_b.dot(theta)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    
    # Plot the training data and the regression line
    plt.plot(X_train, y_train, "b.", label="Training data")
    plt.plot(X_train, X_train_b.dot(theta), "r-", label="Regression line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit (Training Data)")
    plt.legend()
    plt.savefig("plots/linear_regression_fit_training.png")
    plt.show()

    # Plot the test data and the regression line
    plt.plot(X_test, y_test, "g.", label="Test data")
    plt.plot(X_test, X_test_b.dot(theta), "r-", label="Regression line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit (Test Data)")
    plt.legend()
    plt.savefig("plots/linear_regression_fit_test.png")
    plt.show()

    # Plot the cost history
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function History")
    plt.savefig("plots/cost_history.png")
    plt.show()

if __name__ == "__main__":
    main()
