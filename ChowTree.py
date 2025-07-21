from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
import pandas as pd


df = pd.read_csv("synthetic_dataset.csv")

# Step 1: Learn structure using Chow-Liu (TreeSearch with 'chow-liu')
ts = TreeSearch(df)
chow_model = ts.estimate(estimator_type="chow-liu")

# Step 2: Fit CPDs
trained_model = chow_model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')

learned_edges = set(chow_model.edges())

original_edges = {("API_types", 'Granule Size'),
            ('Granulation parameters', 'Granule Size'),
        ('Milling settings', 'Granule Size'),
        ('Blending Settings', 'Blend Uniformity'),
        ('Drying parameters','Moisture Content'),
        ("API_types",'Assay'), 
        ('Compression Settings', 'Tablet Hardness'),
        ('Compression Settings', 'Tablet Weight Variation'),
        ('Coating Parametres','Dissolution Rate'),
        ('Packaging Humidity', 'Dissolution Rate'),
        ('Granule Size', 'Powder Floability'),
        ('Granule Size', 'Tablet Hardness'),
        ('Granule Size', 'Dissolution Rate'),
        ('Blend Uniformity', 'Content Uniformity'),
       ('Moisture Content', 'Dissolution Rate'),
       ('Powder Floability', 'Tablet Weight Variation'),
}

# Compare structures
extra_edges = learned_edges - original_edges
missing_edges = original_edges - learned_edges
correct_edges = learned_edges & original_edges

print(f"Original edges: {len(original_edges)}")
print(f"Learned edges: {len(learned_edges)}")
print(f"Correct edges: {len(correct_edges)}")
print(f"Extra edges (false positives): {len(extra_edges)}")
print(f"Missing edges (false negatives): {len(missing_edges)}")


learned_cpds = {cpd.variable: cpd for cpd in trained_model.get_cpds()}
print("Learned CPDs:")
for var, cpd in learned_cpds.items():
    print(f"CPD of {var}:\n{cpd}\n")

import pickle
with open("chowliu_model.pkl", "wb") as f:
    pickle.dump(trained_model, f)