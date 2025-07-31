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