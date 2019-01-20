# T2D_Models
A reusable machine learning pipeline that can be used for binary, multiclass or multilabel classification. Current output is trained on genetic data where the features are genomic annotations and the target variable is the presence of a particular trait (T2D, Lipids, CAD, BMI, etc). 

### Setup
Clone repository or download .yml file. Ensure anaconda is installed. To install an anaconda environment from a .yml through the command line:

Create environment
```
conda env create -f t2d_model.yml
```

Verify all packages were installed with:
```
conda list
```

Activate anaconda environment
```
source activate t2d_model
```

Once the environment is activated, you can use the jupyter notebooks to run through the workflow.

### Notebooks
*Feature Selection: Configure filepath and then use to create and save feature-selected datasets. Currently configured to store files in an S3 bucket, change this to store to your bucket or a local drive by editing the outpath and replacing write_to_S3 functions with write_to_local. 

*Clustering: Exploratory clustering of data to detect patterns prior to classification

*One Vs Rest Model Evaluation: Runs multiple sklearn models on the training data and compares their performance on test data. Configure filepaths for input data before running. 

*Multilabel Model Evaluation: Similar to One Vs Rest Notebook except for multilabel problems. 

### Classes:
`Evaluator.py` is a helper class for plotting and evaluating model performance. Documentation for methods is contained in the python file. 
