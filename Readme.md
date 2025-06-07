 🧠 Bayesian Quality Control

A project where I'm building a Bayesian Network to model and improve quality control in manufacturing.

This is still a work in progress — I’m experimenting, breaking things, and learning as I go.

## 🚀 What’s This About?

The idea is to apply **Bayesian networks** to understand how different production parameters influence final product quality. Instead of relying only on traditional thresholds or deterministic rules, I want to explore how probabilistic reasoning can help catch issues earlier and more intelligently.

## 🧰 Tools I'm Using

- `pomegranate` – for building the Bayesian network
- `PyTorch` – for modeling side experiments and maybe deeper inference later
- `pandas`, `numpy`, `matplotlib` – the usual data wrangling and plotting crew

## 🗂️ Project Structure (so far)

bayesian-quality-control/
│
├── data/ # Raw and processed data
├── notebooks/ # Jupyter experiments and early drafts
├── src/ # Main logic (model definitions, helpers, etc.)
│ ├── config.py # Configs for reproducibility
│ └── network.py # Building the Bayesian model
├── results/ # Output from experiments, visuals, etc.
├── requirements.txt # Project dependencies
└── README.md # You're here!

## 🎯 What I’m Trying to Answer

- Which manufacturing parameters have the biggest impact on product variation?
- Can a probabilistic model catch hidden patterns better than a rule-based one?
- How reliable are predictions when uncertainty is explicitly modeled?

## ✅ TODO (in progress)

- [x] Define structure of the network
- [x] Prepare synthetic dataset
- [ ] Add real-world dataset
- [ ] Train and evaluate performance
- [ ] Visualize inference & insights

## 🙋‍♂️ About Me

I'm Alexey — I switched from a decade in pharma quality assurance (J&J, Abbott) to data science. I like building things that are both practical and mathematically interesting. This project blends both.

♟️ Also into chess, swimming, and books I never finish.