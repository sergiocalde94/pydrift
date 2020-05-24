.. Data And Model Drift Checker documentation master file, created by
   sphinx-quickstart on Fri May 15 18:54:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Data And Model Drift Checker Docs
=================================

Welcome to the project documentation, each module has its own description and source code that can be found here.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   index

Drift Checker
=============

Core three classes `DriftChecker`, `DataDriftChecker` and `ModelDriftChecker`.

.. automodule:: pydrift.core.drift_checker
   :members:

Interpretable Drift
===================

Core class `InterpretableDrift` for shap interaction and simple plots.

.. automodule:: pydrift.core.interpretable_drift
   :members:


Drift Checker Estimator
=============

Core class `DriftCheckerEstimator`, for sklearn compatibility.

.. automodule:: pydrift.core.drift_checker_estimator
   :members:

Models
======

Sub-module concerning the part of the models.

.. automodule:: pydrift.models
   :members:

Constants
=========

Sub-module concerning the part of the constants.

.. automodule:: pydrift.constants
   :members:

Exceptions
==========

Sub-module concerning the part of the exceptions.

.. automodule:: pydrift.exceptions
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
