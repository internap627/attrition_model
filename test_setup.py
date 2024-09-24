# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset to test if it's being read correctly
try:
    df = pd.read_csv('HumanResources.csv', sep=';')
    print("Dataset loaded successfully.")
    print(df.columns)
except Exception as e:
    print(f"Error loading dataset: {e}")
