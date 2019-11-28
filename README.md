# Anomaly detection in credit card data (Python, scikit-learn) 

Author: [Dennis W. Hallema](https://www.linkedin.com/in/dennishallema) 

Description: Scikit-learn supervised classification procedure for detecting fraudulent credit card transactions in a large dataset. This procedure compares the performance of four classifiers: Logistic Regression, Kernel Support Vector Machine, Stochastic Gradient Boosting and Random Forest. 

Depends: See `environment.yml`. 

Data: PCA transformed credit card transaction data collected in Europe over the course of two days. This anonymized dataset was created by Worldline and the Machine Learning Group of Université Libre de Bruxelles (http://mlg.ulb.ac.be). 

Disclaimer: Use at your own risk. No responsibility is assumed for a user's application of these materials or related materials. 

---

## Cloning this repository

1. Clone this repository onto your machine: 
   `git clone https://github.com/dwhallema/<repo>`, replace `<repo>` with the name of this repository. 
   This creates a new directory "repo" containing the repository files.
2. Install Anaconda and Python  
3. Create a Python environment within the cloned directory using Anaconda: `conda env create`  
   This will download and install the dependencies listed in environment.yml.  
4. Activate the Python environment: `source activate <repo>` (or `conda activate <repo>` on Windows).  
5. Launch the Jupyter Notebook server: `jupyter notebook`  
6. In the opened browser tab, click on a ".ipynb" file to open it.  
7. To run a cell, select it with the mouse and click the "Run" button.  

Troubleshooting: 

* If you need to install dependencies manually run: `conda create -name <repo> dep`  
* If you need to update the Python environment run: `conda env update --file environment.yml`  
