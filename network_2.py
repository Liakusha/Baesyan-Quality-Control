from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch
import pandas as pd
from pomegranate import _utils

# Save the original function
original_check_parameter = _utils._check_parameter

# Monkey patch with debug print
def debug_check_parameter(X, name, ndim=None, shape=None, **kwargs):
    print(f"[DEBUG] {name} shape: {X.shape} | Expected shape: {shape}")
    return original_check_parameter(X, name, ndim=ndim, shape=shape, **kwargs)

_utils._check_parameter = debug_check_parameter

print("✅ Monkey patch applied")

##====Root Nodes====

# API types: 2 categories — Type A, Type B
API_types = Categorical(probs=torch.tensor([[0.4, 0.6]]))

# Granulation parameters: 3 categories — Low, Medium, High
granulation_parameters = Categorical(probs=torch.tensor([[0.35, 0.4, 0.25]]))

# Milling settings: 2 categories — Slow, Fast
milling_parameters = Categorical(probs=torch.tensor([[0.45, 0.55]]))

# Blending Settings: 4 categories Time/Speed - Short/Low, Short/Fast, Long/Low, Long/Fast
blending_settings = Categorical(probs = torch.tensor([[0.2, 0.3, 0.3, 0.2]]))

# Drying parameters: 4 categories Time/Temp - Short/Low, Short/High, Long/Low, Long/High
drying_parameters = Categorical(probs = torch.tensor([[0.15, 0.25, 0.2, 0.4]]))

# Compression Settings: 3 categories - Low/Medium/High Force
compression_settings = Categorical(probs = torch.tensor([[0.3, 0.4, 0.3]]))

# Coating Parametres: 3 categories - Thin/Medium/Thick
coating_parametres = Categorical(probs = torch.tensor([[0.3, 0.5, 0.2]]))

# Packaging Humidity: 3 categories - Low/Medium/High
packaging_humidity = Categorical(probs = torch.tensor([[0.15, 0.75, 0.1]]))

# ====Latent Nodes==== 

# Granule Size: 3 categories — Fine, Medium, Coarse
# Shape: [API][Granulation][Milling][Granule Size]
probs_gran = torch.tensor([
    [  # API = Type A
        [[0.75, 0.20, 0.05], [0.60, 0.30, 0.10]],
        [[0.50, 0.40, 0.10], [0.60, 0.30, 0.10]],
        [[0.30, 0.50, 0.20], [0.15, 0.45, 0.40]],
    ],
    [  # API = Type B
        [[0.65, 0.20, 0.15], [0.50, 0.35, 0.15]],
        [[0.45, 0.45, 0.10], [0.30, 0.55, 0.15]],
        [[0.25, 0.55, 0.20], [0.10, 0.50, 0.40]],
    ],
])

granule_size = ConditionalCategorical(probs=probs_gran)

#Create the distribution list for modeling
distributions_var = [API_types,
                 milling_parameters,
                 blending_settings,
                 granulation_parameters,
                 drying_parameters,
                 compression_settings,
                 coating_parametres,
                 packaging_humidity,
                granule_size]

# Create the edges list
edges_var =[(API_types, granule_size),
            (granulation_parameters, granule_size),
       # (API_types, assay), 
        (milling_parameters, granule_size),
        #(blending_settings, blend_uniformity),
        #(drying_parameters, moisture_content),
        #(compression_settings, tablet_hardness),
        #(compression_settings, tablet_weight_var),
        #(coating_parametres, dissolution_rate),
        #(packaging_humidity, dissolution_rate),
        #(granule_size, powder_floability),
        #(granule_size, tablet_hardness),
        #(granule_size, dissolution_rate),
        #(blend_uniformity, content_unif),
        #(moisture_content, tablet_hardness),
       # (moisture_content, dissolution_rate),
       # (powder_floability, tablet_weight_var),
           ] 

structure_var = [
    (),  # API_types — root
    (),  # granulation_parameters — root
    (),  # milling_parameters — root
    (),  # blending_settings — root
    (),  # drying_parameters — root
    (),  # compression_settings — root
    (),  # coating_parametres — root
    (),  # packaging_humidity — root

    # Latent nodes
    (0, 1, 2),     # granule_size <- API, granulation_parameters, milling_parameters
    #(3,),          # blend_uniformity <- blending_settings
    #(4,),          # moisture_content <- drying_parameters
    #(8,),          # powder_floability <- granule_size

    # Leaf nodes
    #(0,),          # assay <- API_types
    #(9,),          # content_unif <- blend_uniformity
    #(8, 5),        # tablet_hardness <- granule_size, compression_settings
    #(8, 10, 6, 7), # dissolution_rate <- granule_size, moisture_content, coating_parametres, packaging_humidity
    #(5, 11),       # tablet_weight_var <- compression_settings, powder_floability
]
model = BayesianNetwork(distributions=distributions_var, structure= structure_var, edges=edges_var )
#print(probs_gran.size())

sample = model.sample(n = 5)
#print(sample)