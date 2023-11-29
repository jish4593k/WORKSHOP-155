import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

class NeuralNet:
    def __init__(self, dataFile):
        self.raw_input = pd.read_csv(dataFile)

    def preprocess(self):
        # Replaces all categorical data into numerical data
        self.processed_data = self.raw_input.replace(
            ('vhigh', 'high', 'med', 'low', 'big', 'small', '5more', 'more'), (4, 3, 2, 1, 3, 1, 5, 6)
        )

    def create_keras_model(self, activation='relu', learning_rate=0.01, hidden_layer_sizes=(2,)):
        model = keras.Sequential([
            layers.InputLayer(input_shape=(self.processed_data.shape[1] - 1,)),
            layers.Dense(hidden_layer_sizes[0], activation=activation),
            layers.Dense(len(self.processed_data['Class'].unique()), activation='softmax')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Standardize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create a KerasClassifier for use with scikit-learn
        keras_model = KerasClassifier(build_fn=self.create_keras_model, epochs=100, batch_size=32, verbose=0)

        # Use GridSearchCV for hyperparameter tuning
        param_grid = {
            'activation': ['relu', 'tanh'],
            'learning_rate': [0.01, 0.1],
            'hidden_layer_sizes': [(2,), (3,)]
        }
        grid_search = GridSearchCV(keras_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Print the best parameters
        print("Best Parameters: ", grid_search.best_params_)

        # Plot learning curves
        plot_learning_curve(grid_search.best_estimator_, "Learning Curve", X_train, y_train, cv=5)
        plt.show()

        # Evaluate the model on the test set
        test_accuracy = grid_search.score(X_test, y_test)
        print("Test Accuracy: ", test_accuracy)


if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/dboonsu/nn-cars/main/car.csv")
    neural_network.preprocess()
    neural_network.train_evaluate()
