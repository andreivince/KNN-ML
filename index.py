import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Load the car dataset from a CSV file using pandas
data = pd.read_csv("car.data")
print(data.head())

# Encode categorical variables as numeric values using LabelEncoder from preprocessing
le = preprocessing.LabelEncoder() 
buying = le.fit_transform(list(data["buying"])) #Transforming in a numeric value
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
classe = le.fit_transform(list(data["class"]))

# Define the target variable (class) and the features used to predict it (buying, maint, door, persons, lug_boot, safety)
predict = "class"
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(classe)

# Split the dataset into training and testing sets, using 10% of the data for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# Create a KNN model with k=9 neighbors (this value can be adjusted to improve accuracy)
model = KNeighborsClassifier(n_neighbors=9)
# To improve accuracy, experiment with the number of neighbors:
# - n_neighbors = 5 (90% accuracy)
# - n_neighbors = 7 (92% accuracy)
# - n_neighbors = 9 (95% accuracy)

# Fit the model to the training data
model.fit(x_train, y_train)

# Compute the accuracy of the model on the testing data
acc = model.score(x_test, y_test)
print(acc)

# Make predictions on the testing data and print the predicted class, the data, and the actual class
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "very good"]

for x in range(len(predicted)):
    print("\n")
    print("Predicted:", names[predicted[x]])
    print("Data:", x_test[x])
    print("Actual:", names[y_test[x]])
