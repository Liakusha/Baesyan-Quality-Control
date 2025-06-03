from pomegranate.distributions import Categorical, ConditionalCategorical
import torch

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




## ====Latent Nodes==== 

# Granule Size: 3 categories — Fine, Medium, Coarse
# Shape: [API][Granulation][Milling][Granule Size]
probs_gran = [
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
]

granule_size = ConditionalCategorical(probs=torch.tensor(probs_gran))

# Blend Uniformity: 3 categories  - Low, Medium, High
# Shape: [Blending Settings][Blend Uniformity]
probs_blend_unif = [
    [0.6, 0.3, 0.1],  # Short/Slow
    [0.45, 0.3, 0.25],  # Short/Fast
    [0.35, 0.45, 0.2],  # Long/Slow
    [0.3, 0.5, 0.2],  # Long/Fast
]

blend_uniformity = ConditionalCategorical(probs=[torch.tensor(probs_blend_unif)])

# Moisture Content: 3 categories - Low, Medium,High
# Shape: [Drying Parameters][Moisture Content]

probs_moisture = [
   [0.1, 0.2, 0.7], #Short/Low
   [0.15, 0.4, 0.45], #Short/High
   [0.35, 0.35, 0.3], #Long/Low
   [0.5, 0.4, 0.1], #Long/High
]
moisture_content = ConditionalCategorical(probs = [torch.tensor(probs_moisture)])

# Powder Floability: 2 Categories - Poor, Good
# Shape [Granule Size][Powder Floability]
probs_powder = [
    [0.75, 0.25], #Fine
    [0.5, 0.5], #Medium
    [0.25, 0.75], #Low
]
powder_floability = ConditionalCategorical(probs = [torch.tensor(probs_powder)])
                                           

## ====Leaf Nodes====

#Assay: 3 categorie - Low, Acceptable, High
#Shape [API type][Assay]
probs_assay = [
    [0.1, 0.8, 0.1], #API A
    [0.15, 0.65, 0.2], #API B
]
assay = ConditionalCategorical(probs = [torch.tensor(probs_assay)])

#Content Uniformity: 3 categories -  Low, Medium, High
#Shape [Blend Uniformity][Content Uniformity]
probs_content_unif = [
    [0.6, 0.3, 0.1], #Low
    [0.35, 0.5, 0.15], #Medium
    [0.2, 0.35, 0.45], #High
]
content_unif = ConditionalCategorical(probs = [torch.tensor(probs_content_unif)])

#Tablet Hardness: 3 categories - Low, Medium, High
#Shape [Granule Size][Compresion Settings][Tablet Hardness]

probs_tablet_hard = [
    #Fine Granule Size
    [[0.15, 0.75, 0.1], [0.13, 0.76, 0.11], [0.11, 0.75, 0.14]],
    #Medium Granule Size
    [[0.25, 0.55, 0.2], [0.23, 0.56, 0.21], [0.21, 0.55, 0.24]],
    #Coarse Granule Size
    [[0.55, 0.42, 0.03], [0.53, 0.43, 0.04], [0.52, 0.42, 0.06]],
]
tablet_hardness = ConditionalCategorical(probs = [torch.tensor(probs_tablet_hard)])

# Dissolution Rate: 3 categories - Slow, Accebtable, Fast
#Shape: [Granule Size][Moisture Content][Coating Parameters][Packaging Humidity][Dissolution Rate]
probs_dissolution = torch.tensor([
    [  # Fine
        [  # Moisture: Low, Med, High
            [  # Coating: Thin, Medium, Thick
                [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.4, 0.4, 0.2]],  # Packaging Humidity: Low, Med, High
                [[0.55, 0.35, 0.1], [0.45, 0.45, 0.1], [0.4, 0.4, 0.2]],
                [[0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.35, 0.45, 0.2]],
            ],
        ],
        [
            [
                [[0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.3, 0.5, 0.2]],
                [[0.45, 0.45, 0.1], [0.35, 0.55, 0.1], [0.3, 0.5, 0.2]],
                [[0.4, 0.5, 0.1], [0.3, 0.6, 0.1], [0.25, 0.55, 0.2]],
            ],
        ],
        [
            [
                [[0.4, 0.5, 0.1], [0.3, 0.6, 0.1], [0.2, 0.6, 0.2]],
                [[0.35, 0.55, 0.1], [0.25, 0.65, 0.1], [0.2, 0.6, 0.2]],
                [[0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.15, 0.65, 0.2]],
            ],
        ],
    ],
    [  # Medium
        [
            [
                [[0.4, 0.4, 0.2], [0.3, 0.5, 0.2], [0.25, 0.5, 0.25]],
                [[0.35, 0.45, 0.2], [0.25, 0.55, 0.2], [0.2, 0.5, 0.3]],
                [[0.3, 0.5, 0.2], [0.2, 0.6, 0.2], [0.15, 0.55, 0.3]],
            ],
        ],
        [
            [
                [[0.3, 0.5, 0.2], [0.2, 0.6, 0.2], [0.15, 0.55, 0.3]],
                [[0.25, 0.55, 0.2], [0.15, 0.65, 0.2], [0.1, 0.6, 0.3]],
                [[0.2, 0.6, 0.2], [0.1, 0.7, 0.2], [0.05, 0.65, 0.3]],
            ],
        ],
        [
            [
                [[0.2, 0.6, 0.2], [0.1, 0.7, 0.2], [0.05, 0.65, 0.3]],
                [[0.15, 0.65, 0.2], [0.05, 0.75, 0.2], [0.03, 0.65, 0.32]],
                [[0.1, 0.7, 0.2], [0.02, 0.78, 0.2], [0.01, 0.7, 0.29]],
            ],
        ],
    ],
    [  # Coarse
        [
            [
                [[0.7, 0.25, 0.05], [0.65, 0.3, 0.05], [0.6, 0.3, 0.1]],
                [[0.65, 0.3, 0.05], [0.6, 0.35, 0.05], [0.55, 0.35, 0.1]],
                [[0.6, 0.3, 0.1], [0.55, 0.35, 0.1], [0.5, 0.35, 0.15]],
            ],
        ],
        [
            [
                [[0.6, 0.3, 0.1], [0.55, 0.35, 0.1], [0.5, 0.35, 0.15]],
                [[0.55, 0.35, 0.1], [0.5, 0.4, 0.1], [0.45, 0.4, 0.15]],
                [[0.5, 0.4, 0.1], [0.45, 0.45, 0.1], [0.4, 0.45, 0.15]],
            ],
        ],
        [
            [
                [[0.5, 0.4, 0.1], [0.45, 0.45, 0.1], [0.4, 0.45, 0.15]],
                [[0.45, 0.45, 0.1], [0.4, 0.5, 0.1], [0.35, 0.5, 0.15]],
                [[0.4, 0.5, 0.1], [0.35, 0.5, 0.15], [0.3, 0.5, 0.2]],
            ],
        ],
    ]
])

dissolution_rate = ConditionalCategorical(probs=torch.tensor(probs_dissolution))

# Tablet Weight Variation: 3 categories - Low, Medium, High
# Shape: [Compression Settings][Powder Floability][Tablet Weight Variation]
probs_tablet_weight = [
    # Good Flowability
    [[0.15, 0.75, 0.1], [0.13, 0.76, 0.11], [0.11, 0.75, 0.14]],
    # Poor Flowability
    [[0.25, 0.55, 0.2], [0.23, 0.56, 0.21], [0.21, 0.55, 0.24]],
]
tablet_weight_var = ConditionalCategorical(probs=torch.tensor(probs_tablet_weight))