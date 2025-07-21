from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

#Open the model
with open ("manuf_baeysian_model.pkl", "rb") as f:
    model = pickle.load(f)

#List of nodes
nodes_name = ["API types",
              "Granulation parameters",
              "Milling settings",
              "Blending Settings",
              "Drying parameters", 
              "Compression Settings",
              "Coating Parametres",
              "Packaging Humidity",
              "Granule Size",
              "Blend Uniformity",
              "Moisture Content",
              "Powder Floability",
              "Assay",
              "Content Uniformity",
              "Tablet Hardness",
              "Dissolution Rate",
              "Tablet Weight Variation"]

#List of root nodes
root_nodes = ["API types",
              "Granulation parameters",
              "Milling settings",
              "Blending Settings",
              "Drying parameters", 
              "Compression Settings",
              "Coating Parametres",
              "Packaging Humidity"
              ]


#Map our nodes in readeble format
value_maps = {"API types": {0:"Type A", 1: "Type B"},
              "Granulation parameters":  {0:"Low", 1 : "Medium", 2 : "High"},
              "Milling settings": {0: "Slow", 1 : "Fast"},
              "Blending Settings" : {0: "Short/Low", 1 : "Short/Fast", 2 : "Long/Low", 3: "Long/Fast"},
              "Drying parameters" : {0: "Short/Low", 1 : "Short/High", 2: "Long/Low", 3 : "Long/High"},
              "Compression Settings" : {0 : "Low", 1 : "Medium", 2 : "High"},
              "Coating Parametres" : {0 : "Thin", 1 : "Medium", 2 : "Thick"},
              "Packaging Humidity" :  {0 : "Low", 1 : "Medium", 2 : "High"},
              "Granule Size" : {0: "Fine", 1 : "Medium", 2 : "Coarse"},
              "Blend Uniformity" : {0 : "Low", 1 : "Medium", 2 : "High"},
              "Moisture Content" : {0 : "Low", 1 : "Medium", 2 : "High"},
              "Powder Floability" : {0 :"Poor",1 : "Good"},
              "Assay" : {0 : "Low", 1 : "Acceptable", 2: "High"},
              "Content Uniformity" : {0 : "Low", 1 : "Medium", 2 : "High"},
              "Tablet Hardness" : {0 : "Low", 1 : "Medium", 2 : "High"},
              "Dissolution Rate" : {0 : "Slow", 1 : "Accebtable", 2: "Fast"},
              "Tablet Weight Variation" : {0 : "Low", 1 : "Medium", 2 : "High"},
                     }

# Set reproducible seed for PyTorch
torch.manual_seed(42)

# Quantity of samples
N_SAMPLES = 50000

#Raw sample
raw_sample = model.sample(N_SAMPLES)

# Present our samples as DataFrame in readble format 
samples = pd.DataFrame(raw_sample.numpy(), columns=[f'Node_{i}' for i in range(len((model.sample(N_SAMPLES))[0]))])
samples.columns = nodes_name
df = samples.apply(lambda col: col.replace(value_maps[col.name]))
df.to_csv("synthetic_dataset.csv", index=False)


