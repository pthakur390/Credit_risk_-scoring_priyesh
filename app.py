import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Train a simple model inside the app
X, y = make_classification(n_samples=500, n_features=5, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
