import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name']
df = pd.read_csv(url, sep='\s+', names=column_names, na_values="?")

# Remove missing values
df = df.dropna()

# Convert MPG to liters per 100 km
df['Fuel_Consumption_L_per_100km'] = 235.215 / df['MPG']

# Create target variable: Economical (0) if < 10 L/100 km, Gas Guzzler (1) if >= 10 L/100 km
df['Efficiency'] = (df['Fuel_Consumption_L_per_100km'] >= 10).astype(int)

# Prepare features
X = df[['Horsepower', 'Weight', 'Cylinders']].values  # Using Horsepower, Weight, and Cylinders
y = df['Efficiency'].values

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Decision Tree)
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limit depth for simplicity
model.fit(X_train, y_train)
print("\n=== Car Efficiency Classification Results (Decision Tree) ===")
print("Accuracy on training data:", model.score(X_train, y_train))
print("Accuracy on test data:", model.score(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)
print("First 5 predictions (0=Economical, 1=Gas Guzzler):", y_pred[:5])
print("Actual values:", y_test[:5])

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['Horsepower', 'Weight', 'Cylinders'], class_names=['Economical', 'Gas Guzzler'], filled=True)
plt.title("Decision Tree for Car Efficiency")
plt.savefig("decision_tree.png")  # Save for GitHub
plt.show()

# Additional visualization: Fuel Consumption vs Horsepower with color by efficiency
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Horsepower'], df['Fuel_Consumption_L_per_100km'], c=df['Efficiency'], cmap='coolwarm', alpha=0.6)
plt.xlabel("Horsepower")
plt.ylabel("Fuel Consumption (L/100 km)")
plt.title("Fuel Consumption vs Horsepower by Efficiency")
plt.colorbar(scatter, label='Efficiency (0=Economical, 1=Gas Guzzler)')
plt.grid(True)
plt.savefig("efficiency_scatter.png")  # Save for GitHub
plt.show()
