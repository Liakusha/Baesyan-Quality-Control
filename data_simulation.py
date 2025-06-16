import numpy as np
import pandas as pd
import random

API_types = ["Type A", "Type B"] 

granulation_parameters = ["Low", "Medium", "High"]

milling_parameters = ["Slow", "Fast"]

blending_settings = ["Short/Low", "Short/Fast", "Long/Low", "Long/Fast"]

drying_parameters = ["Short/Low", "Short/High", "Long/Low", "Long/High"]

compression_settings = ["Low", "Medium", "High"]

coating_parametres = ["Thin","Medium","Thick"]
 
packaging_humidity =["Low","Medium","High"]

granule_size= ["Fine", "Medium", "Coarse"]

blend_uniformity = ["Low", "Medium", "High"]

moisture_content = ["Low","Medium","High"]

powder_floability = ["Poor", "Good"]

assay = ["Low", "Acceptable", "High"]

content_unif = ["Low","Medium","High"]

tablet_hardness = ["Low","Medium","High"]

dissolution_rate = ["Slow", "Accebtable", "Fast"]

tablet_weight_var = ["Low","Medium","High"]

num_samples = 1000

data = {
    "API_types": random.choices(API_types, k=num_samples),
    "granulation_parameters": random.choices(granulation_parameters, k=num_samples),
    "milling_parameters": random.choices(milling_parameters, k=num_samples),
    "blending_settings": random.choices(blending_settings, k=num_samples),
    "drying_parameters": random.choices(drying_parameters, k=num_samples),
    "compression_settings": random.choices(compression_settings, k=num_samples),
    "coating_parametres": random.choices(coating_parametres, k=num_samples),
    "packaging_humidity": random.choices(packaging_humidity, k=num_samples),
    "granule_size": random.choices(granule_size, k=num_samples),
    "blend_uniformity": random.choices(blend_uniformity, k=num_samples),
    "moisture_content": random.choices(moisture_content, k=num_samples),
    "powder_floability": random.choices(powder_floability, k=num_samples),
    "assay": random.choices(assay, k=num_samples),
    "content_unif": random.choices(content_unif, k=num_samples),
    "tablet_hardness": random.choices(tablet_hardness, k=num_samples),
    "dissolution_rate": random.choices(dissolution_rate, k=num_samples),
    "tablet_weight_var": random.choices(tablet_weight_var, k=num_samples),
}

df =pd.DataFrame(data)
print(df.head())

from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
encoded_data = encoder.fit_transform(df)
encoded_df = pd.DataFrame(encoded_data, columns=df.columns)
print(encoded_df.head())