from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from scipy.stats import entropy

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


# Quantity of samples
N_SAMPLES = 10

# Present our samples as DataFrame in readble format 
samples = pd.DataFrame((model.sample(N_SAMPLES)).numpy(), columns=[f'Node_{i}' for i in range(len((model.sample(N_SAMPLES))[0]))])
samples.columns = nodes_name
samples = samples.apply(lambda col: col.replace(value_maps[col.name]))

#node_idx = nodes_name.index("API types")
#true_dist = model.distributions[node_idx]
#params = list(true_dist.parameters())
#probs_tensor = params[1]
#labels = list(value_maps["API types"].values())
#true_probs = dict(zip(labels, probs_tensor.squeeze().tolist()))

# Perform the true probabilities from the model
true_distributions = {}

for i, node_name in enumerate(nodes_name):
    dist = model.distributions[i]
    params = list(dist.parameters())
    probs_tensor = params[1]
    labels = list(value_maps[node_name].values())
    true_probs = dict(zip(labels, probs_tensor.squeeze().tolist()))
    true_distributions[node_name] = true_probs
    
 # Calculate the emperical probabilities based on sample data

empirical_distributions = {}
for node in nodes_name:
    empirical_probs = samples[node].value_counts(normalize=True).to_dict()
    empirical_distributions[node] = empirical_probs

#Plot comparison between true and emperical probabilities
def compare_distributions(node, true_probs, emp_probs):
    #Set of labels
    all_labels = sorted(set(true_probs) | set(emp_probs))

    #Vectors of probabilities
    true_vec = [true_probs.get(k,0.0) for k in all_labels]
    emp_vec = [emp_probs.get(k, 0.0) for k in all_labels]

    #Metrics
    l1 = np.sum(np.abs(np.array(true_vec) - np.array(emp_vec)))
    kl = entropy(pk=true_vec, qk=emp_vec) if all(p>0 for p in emp_vec) else float("NaN")

    #Plot
    x = np.arange(len(all_labels))
    plt.bar(x - 0.2, true_vec,  width=0.4, label='True', alpha=0.7)
    plt.bar(x + 0.2, emp_vec,   width=0.4, label='Empirical', alpha=0.7)
    plt.xticks(x, all_labels, rotation=45)
    plt.title(f"{node} | L1 = {l1:.3f}, KL = {kl:.3f}")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

summary = []

for node in nodes_name:
    labels = sorted(set(true_distributions[node].keys()) | set(empirical_distributions[node].keys()))
    t_vec = [true_distributions[node].get(k, 0) for k in labels]
    e_vec = [empirical_distributions[node].get(k, 0) for k in labels]
    
    #l1 = np.sum(np.abs(np.array(t_vec) - np.array(e_vec)))
    #kl = entropy(pk=t_vec, qk=e_vec) if all(p > 0 for p in e_vec) else float('nan')
    ##summary.append((node, round(l1, 4), round(kl, 4)))

#for row in summary:
    #print(f"{row[0]:<25} | L1: {row[1]:.4f} | KL: {row[2]:.4f}")
print("t_vec:", t_vec)
print("e_vec:", e_vec)