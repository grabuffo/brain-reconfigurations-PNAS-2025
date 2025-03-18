# Project Overview

This project involves generating simulated training time series, extracting statistical features, 
estimating the $G$ and 6 $\eta$, and performing posterior predictive checks to find the optimal parameters for fitting the empirical fMRI recordings.

## Scripts Overview

### `run.py`
This script generates 100,000 training simulation time series by providing priors for $G$ (global coupling strength) and 6 $eta$ (1 $\eta$ for each subnetwork). It creates and submits jobs to a given cluster by calling `one_batch.py`.

### `script_extract_features.py`
This script loads the simulated time series and extracts statistical features from the Functional Connectivity (FC) and Functional Connectivity Dynamics (FCD) matrices for each simulated training time series.

### `note_infer.ipynb`
Use the extracted features, apply parameter estimation ans store estimated parameters.

### `note.ipynb`
This notebook is used for analyzing posterior predictive checks and identifying the best fit based on the estimated parameters $G$ and $eta$ for the 6 subnetworks.