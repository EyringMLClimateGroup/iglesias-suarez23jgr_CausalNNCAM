# CausalNNCAM - Causally-informed deep learning to improve climate models and projections  

Authors:
* Fernando Iglesias-Suarez - fernando.iglesias-suarez@dlr.de
* Breixo Soliño Fernández - breixo.solinofernandez@dlr.de
* Original [CBRAIN-CAM](https://github.com/raspstephan/CBRAIN-CAM) code by Stephan Rasp - raspstephan@gmail.com - https://raspstephan.github.io

This repository provides the source code used on the paper [Causally-informed deep learning to improve climate models and projections](https://meetingorganizer.copernicus.org/EGU23/EGU23-6450.html), by Iglesias-Suarez et al.

## Installation

To install the dependencies, it is recomended to use Anaconda or Mamba. An environment file is provided in `dependencies.yml`.

## How to reproduce results

The results described in the paper where obtained by execute the following steps:

1. `aggregate_results.ipynb`: Collect, aggregate and evaluate causal links. Produces causal (correlation) matrix plots.
2. Find the appropriate threshold to filter spurious links, using SHERPA.
    1. `SHERPA_threshold_GridSearch.ipynb`: Best general threshold.
    2. `notebooks_SHERPA_thrs_optimization_per_output/Create_optimized_numparents_dict_mse.ipynb`: Best threshold for each output.
3. Creation and training of neural networks (NN).
    1. `NN_Creation.ipynb`: Can create both Causally-informed NN that use the best general threshold (from 2.1) and Non-causal NN that use all inputs.
    2. `NN_Creation_optimized_threshold.ipynb`: Causally-informed NN that use the best threshold for each output (from 2.2).
    3. `NN_Creation_random_links.ipynb`: NN using random links
4. Evaluation
    1. `notebooks_evaluate_CausalNNs_r2/evaluate_nonlinearities_in_SPCAM.ipynb`: Comparation between the different types of NN
    2. `notebooks_online_evaluation`: Comparation with SPCAM
        1. `cross_section_online_evaluation.ipynb`
        2. `latitudinal_2Dfields_online_evaluation.ipynb`
5. `notebooks_xai/shap_xai.ipynb`: Use explainable AI to evaluate the importance of the inputs in each NN
