# Project Overview

The goal of this project is to estimate 1 $\eta$ for the target regions in different groups of subjects. This involves generating simulated training time series, extracting statistical features, and analyzing posterior predictive checks to identify the best fit for the estimated parameter $\eta$.

## Scripts Overview

### `run.py`
This script provides simulated training time series by generating priors for $\eta$ for the target regions.

### `script_extract_features.py`
This script loads the simulated time series and extracts statistical features from the Functional Connectivity (FC) and Functional Connectivity Dynamics (FCD) matrices for each simulated training time series.

### `note.ipynb`
This notebook is used for analyzing posterior predictive checks and identifying the best fit based on the estimated parameter $\eta$ for the target regions.