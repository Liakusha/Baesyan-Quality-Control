import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pgmpy.estimators import BIC, K2, HillClimbSearch

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

#Best model
best_model = estimator.estimate(scoring_method=scoring_method)

from pgmpy.estimators import BayesianEstimator

trained_model = best_model.fit(df, estimator=BayesianEstimator, prior_type = "BDeu", equivalent_sample_size = 10)

learned_edges = set(best_model.edges())

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
with open("HillClimb+BIC.pkl", "wb") as f:
    pickle.dump(trained_model, f)