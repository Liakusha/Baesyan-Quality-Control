import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pgmpy.estimators import BIC, K2, HillClimbSearch
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork
from pgmpy.estimators.CITests import chi_square
import pickle
import matplotlib.pyplot as plt
import networkx as nx

original_edges = {("API types", 'Granule Size'),
            ('Granulation parameters', 'Granule Size'),
        ('Milling settings', 'Granule Size'),
        ('Blending Settings', 'Blend Uniformity'),
        ('Drying parameters','Moisture Content'),
        ("API types",'Assay'), 
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

with open ("manuf_baeysian_model.pkl", "rb") as f:
    original_model = pickle.load(f)
with open ("chowliu_model.pkl", "rb") as f:
    tree_model = pickle.load(f)
with open ("HillClimb+BIC.pkl", "rb") as f:
    HillBIC_model = pickle.load(f)
with open ("PC_model.pkl", "rb") as f:
    PC_model = pickle.load(f)

#Ectract edges from each model
def get_model_edges(model):
    return set(model.edges())

tree_edges = get_model_edges(tree_model)
hillbic_edges = get_model_edges(HillBIC_model)
pc_edges = get_model_edges(PC_model)
original_edges = set(original_edges)

#Let`s compare the sructures:
# True Positives (TP): edges present in both models 
# False Positives (FP): extra edges in the learned model 
# False Negatives (FN): missing edges that exist in the original

#Define function for comparison
def compare_edges(learned_edges, original_edges):
    tp = learned_edges & original_edges
    fp = learned_edges - original_edges
    fn = original_edges - learned_edges
    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }
#Compare structures
tree_cmp = compare_edges(tree_edges, original_edges)
hillbic_cmp = compare_edges(hillbic_edges, original_edges)
pc_cmp = compare_edges(pc_edges, original_edges)

#Print summary
for name, cmp in zip(["Chow-Liu", "HillClimb+BIC", "PC"], [tree_cmp, hillbic_cmp, pc_cmp]):
    print(f"\nModel: {name}")
    print(f"  Precision: {cmp['Precision']:.2f}")
    print(f"  Recall:    {cmp['Recall']:.2f}")
    print(f"  F1 Score:  {cmp['F1-score']:.2f}")
    print(f"  Extra Edges (FP): {cmp['FP']}")
    print(f"  Missing Edges (FN): {cmp['FN']}")

