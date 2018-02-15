# Deep Learning (Keras) Models Deployment using SQL databases

Are you curious to know if a SQL database can be used to deploy/evaluate a deep-learning model instead of the standard CPU/GPU/CUDA/OpenCL machinery ?

We use Sklearn2sql. Sklearn2sql provides a framework for translating scikit-learn predictive models into a SQL code for deployment purposes. Using this framework, for example, it is possible for a C, perl or java developper to deploy such a model simply by executing the [generated SQL code](https://github.com/antoinecarme/sklearn2sql-demo/blob/master/sample_outputs_round_4/MLPClassifier/BreastCancer/oracle/demo1_MLPClassifier_oracle.sql). The system supports the major market databases.

The goal of this POC is to see if this framework can be applied to deep learning models (keras + scikit-learn wrapper).


In a first step, we investigate the SQL code generation for basic deep learning models ([keras core layers](https://keras.io/layers/core/) and [activation functions](https://keras.io/layers/advanced-activations/)). A second step will investigate basic convolutional models (with [convolutional](https://keras.io/layers/convolutional/) and [pooling](https://keras.io/layers/pooling/) layers).

We are aware that deep learning models tend to have a large number of parameters (layer weights) and hope that SQL deployment wil be usable for small and medium models. 

An evaluation of database capabilities with respect to the model size is already a real-world assessment of this task. 

An additional/optional path to explore is to evaluate SQL code generation for the family of recursive models (RNN , LSTM and GRU, etc) and more advanced keras features.

Your feedback is welcome.
