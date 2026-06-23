# Transformer for DarkPhoton Identification
A repository for investigating Dark Photon identifications with Transformer architectures


## TL;DR
Dark photons are hypothetical beyond-Standard-Model particles whose calorimeter signatures are sparse, irregular, and difficult to distinguish from QCD background. We represent jet energy deposits as graphs and apply Graph Transformer architectures — extended with a Mixture of Experts (MoE) mechanism — to identify displaced dark-photon decays in the ATLAS calorimeter, benchmarking against CNN, GCN, and GAT baselines.

## Data 
The dataset is based on publicly released Monte Carlo simulations of the ATLAS detector, accessible via [HEPData](https://www.hepdata.net/record/ins2728869). A preprocessed version used for training is available at [this link](https://github.com/alessiodevoto/darkphoton)

## Metrics 
We train and evaluate the model and compare it with other architectures (Multilayer perception, Graph Convolutional Neural network, Graph Transformer and Convolutional Neural Network). The results are shown in the table \
<img src="images/metrics.png" alt="Metrics" width="800"/>

## Visualization of Physical Quantities
We analyse physics-motivated variables across prediction categories (TP, TN, FP, FN) to interpret 
model behaviour. The plots show the correlation between reclustered jet $p_T$ and $e_{ratio}$, 
and between $e_{ratio}$ and the number of graph nodes, for each prediction category.

<p float="left">
  <img src="images/corr_pt.png" alt="pT vs eratio" width="400" height="300"/>
  <img src="images/corr_eratio.png" alt="eratio vs nodes" width="400" height="300"/>
</p>


