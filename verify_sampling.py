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


# Quantity of samples
N_SAMPLES = 500

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

# Extract true probabilities from the model for root nodes
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

#Plot comparison between true and emperical probabilities for root nodes
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

node = "Blending Settings"
true = true_distributions[node]
emp = empirical_distributions[node]
#compare_distributions(node, true, emp)

# Create summary for comaprison for root nodes
summary = []

for node in root_nodes:
    labels = sorted(set(true_distributions[node].keys()) | set(empirical_distributions[node].keys()))
    t_vec = [true_distributions[node].get(k, 0) for k in labels]
    e_vec = [empirical_distributions[node].get(k, 0) for k in labels]
    
    l1 = np.sum(np.abs(np.array(t_vec) - np.array(e_vec)))
    kl = entropy(pk=t_vec, qk=e_vec) if all(p > 0 for p in e_vec) else float('nan')
    summary.append((node, round(l1, 4), round(kl, 4)))

#for row in summary:
    #print(f"{row[0]:<25} | L1: {row[1]:.4f} | KL: {row[2]:.4f}")

#Function to find parents for nodes
def get_parents(node_name, nodes_name, structure):
    node_idx = nodes_name.index(node_name)
    parent_indices = structure[node_idx]
    return [nodes_name[i] for i in parent_indices]

from collections import defaultdict

# Function to calculate emperical probs in cond nodes
def compute_empirical_cpd(samples, target_node, parents):
    grouped = samples.groupby(parents)[target_node].value_counts(normalize=True)
    grouped = grouped.rename("prob").reset_index()
    
    result = defaultdict(dict)
    
    for _, row in grouped.iterrows():
        parent_vals = tuple(row[p] for p in parents)
        target_val = row[target_node]
        prob = row["prob"]
        result[parent_vals][target_val] = prob
    
    return result

#Function to extract probs from model
def extract_true_conditional_cpd(node_name, model, nodes_name, value_maps):
    idx = nodes_name.index(node_name)
    dist = model.distributions[idx]
    probs = list(dist.parameters())[1]  # still a tensor, probably shape (3, 3, 3)

    parent_names = get_parents(node_name, nodes_name, model.structure)
    parent_categories = [list(value_maps[parent].values()) for parent in parent_names]
    child_categories = list(value_maps[node_name].values())

    from itertools import product
    combinations = list(product(*parent_categories))

    # Flatten probs to match combinations
    probs_flat = list(probs.reshape(len(combinations), -1))  # shape (9, 3)

    result = {}
    for parent_vals, child_probs in zip(combinations, probs_flat):
        result[parent_vals] = dict(zip(child_categories, child_probs.tolist()))
    return result

true_cpd_granul_size = extract_true_conditional_cpd("Dissolution Rate", model, nodes_name, value_maps)
emp_cpd_granul_size = compute_empirical_cpd(samples, "Dissolution Rate", get_parents("Dissolution Rate", nodes_name, model.structure) )


#function to compare conditional distribution
def compare_conditional_distributions(node, true_cpd, empirical_cpd):
    summary = []
    
    for parent_values in true_cpd:
        true_dist = true_cpd[parent_values]
        emp_dist = empirical_cpd.get(parent_values, {})
        
        all_labels = sorted(set(true_dist) | set(emp_dist))
        
        t_vec = [true_dist.get(k, 0.0) for k in all_labels]
        e_vec = [emp_dist.get(k, 0.0) for k in all_labels]
        
        l1 = np.sum(np.abs(np.array(t_vec) - np.array(e_vec)))
        kl = entropy(pk=t_vec, qk=e_vec) if all(p > 0 for p in e_vec) else float("nan")
        
        summary.append((parent_values, round(l1, 4), round(kl, 4)))
    
    return summary

print(compute_empirical_cpd(samples, "Dissolution Rate", get_parents("Dissolution Rate", nodes_name, model.structure)))


