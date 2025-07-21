from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork
from pgmpy.estimators.CITests import chi_square
import pandas as pd

# Load your dataset
df = pd.read_csv("synthetic_dataset.csv")  

from pgmpy.estimators import PC


pc = PC(data=df)

model = pc.estimate(
    ci_test=chi_square,
    significance_level=0.01
)

learned_edges = set(model.edges())

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

import pickle
with open("PC_model.pkl", "wb") as f:
    pickle.dump(model, f)