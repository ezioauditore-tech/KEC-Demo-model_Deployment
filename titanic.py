# titanic.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    return data

def preprocess_data(data):
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
    data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
    return data

def split_data(data):
    processed_data = preprocess_data(data)
    X = processed_data.drop('Survived', axis=1)
    y = processed_data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_filename="titanic_model.pkl"):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the model as a pickle file
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

if __name__ == "__main__":
    # Load the Titanic dataset
    titanic_data = load_data()
    X_train, X_test, y_train, y_test = split_data(titanic_data)

    # Train the model and save as a pickle file
    trained_model = train_model(X_train, y_train, model_filename="titanic_model.pkl")

    # Evaluate the model
    test_accuracy = evaluate_model(trained_model, X_test, y_test)
    print(f"Testing Accuracy: {test_accuracy:.2f}")
