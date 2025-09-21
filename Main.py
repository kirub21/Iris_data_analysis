import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load Iris dataset from sklearn and convert to DataFrame
try:
    iris_raw = load_iris(as_frame=True)
    iris = iris_raw.frame
    print("Iris dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first few rows
print("First five rows of the dataset:")
print(iris.head())

# Check data types and missing values
print("\nData types:")
print(iris.dtypes)

print("\nMissing values per column:")
print(iris.isnull().sum())

# Clean the dataset by dropping/filling missing values (Iris has none, but let's show the code)
if iris.isnull().any().any():
    iris = iris.dropna()
    print("Missing values dropped.")
else:
    print("No missing values found.")

# Task 2: Basic Data Analysis

# Statistics for numerical columns
print("\nBasic statistics for numerical columns:")
print(iris.describe())

# Group by 'target' (species) and compute mean of numerical columns
species_map = dict(zip(range(3), iris_raw.target_names))
iris['species'] = iris['target'].map(species_map)

grouped = iris.groupby('species').mean(numeric_only=True)
print("\nMean of numerical columns grouped by species:")
print(grouped)

# Identify patterns/interesting findings
print("\nInteresting Findings:")
for col in iris.columns[:-2]:  # Exclude 'target' and 'species'
    max_species = grouped[col].idxmax()
    min_species = grouped[col].idxmin()
    print(f"- For '{col}', '{max_species}' has the highest mean, '{min_species}' the lowest.")

# Task 3: Data Visualization

sns.set(style="whitegrid")

# 1. Line chart: Simulate a "trend" using row index as "time"
plt.figure(figsize=(8, 4))
for species in iris['species'].unique():
    plt.plot(iris[iris['species'] == species].index, 
             iris[iris['species'] == species]['sepal length (cm)'],
             label=species)
plt.title('Sepal Length Trend per Species (Index as Time)')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart: Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'], palette='pastel')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of Sepal Width
plt.figure(figsize=(6, 4))
for species in iris['species'].unique():
    sns.histplot(iris[iris['species'] == species]['sepal width (cm)'], 
                 label=species, kde=True, alpha=0.5, bins=12)
plt.title('Distribution of Sepal Width by Species')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Scatter plot: Sepal Length vs Petal Length colored by Species
plt.figure(figsize=(6, 5))
sns.scatterplot(data=iris, x='sepal length (cm)', y='petal length (cm)', 
                hue='species', palette='deep', s=70)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# End
print("\nAnalysis and visualizations complete. All plots are customized and labeled.")
