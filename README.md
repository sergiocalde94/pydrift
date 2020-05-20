# Welcome to `pydrift` 0.1.6

How do we measure the degradation of a machine learning process? Why does the performance of our predictive models decrease? Maybe it is that a data source has changed (one or more variables) or maybe what changes is the relationship of these variables with the target we want to predict. `pydrift` tries to facilitate this task to the data scientist, performing this kind of checks and somehow measuring that degradation.

# Install `pydrift` :v:

`pip install pydrift`

# Structure :triangular_ruler:

This is intended to be user-friendly. pydrift is divided into **DataDriftChecker** and **ModelDriftChecker**:

- **DataDriftChecker**: search for drift in the variables, check if their distributions have changed
- **ModelDriftChecker**: search for drift in the relationship of the variables with the target, checks that the model behaves the same way for both data sets

Both can use a discriminative model (defined by parent class **DriftChecker**), where the target would be binary in belonging to one of the two sets, 1 if it is the left one and 0 on the contrary. If the model is not able to differentiate given the two sets, there is no difference!

![Class inheritance](/images/class_inheritance.png)

# Usage :book:

You can take a look to the `notebooks` folder where you can find one example for `DataDriftChecker` and other one for `ModelDriftChecker`. 

For more info check the docs available [here](https://sergiocalde94.github.io/Data-And-Model-Drift-Checker/)

More demos and code improvements will coming, if you want to contribute you can contact me, in the future I will upload a file to explain how this would work.
