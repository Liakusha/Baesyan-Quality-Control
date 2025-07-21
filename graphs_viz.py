import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pgmpy.estimators import BIC, HillClimbSearch

# Load your dataset
df = pd.read_csv("synthetic_dataset.csv")

# Encode all categorical columns to integers
encoders = {}  # Save the mappings to decode later

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save the encoder for later reverse mapping if needed

#Score and Estimator
scoring_method = BIC(df)
estimator = HillClimbSearch(df)

#model
best_model = estimator.estimate(scoring_method=scoring_method)

# Create the graph
G = nx.DiGraph(best_model.edges())

# Use kamada_kawai_layout to avoid overlap
pos = nx.kamada_kawai_layout(G)

# Plot
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrowsize=40, font_size=10)
plt.title("Learned Bayesian Network Structure")
plt.show()