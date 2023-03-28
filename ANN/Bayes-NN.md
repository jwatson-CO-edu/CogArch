# Hands-on Bayesian Neural Networks – A Tutorial for Deep Learning Users  
1. https://arxiv.org/pdf/2007.06823.pdf

### Motivation
* DNN Issues
    - Prone to overfitting
    - Overconfident predictions
    - Confidence is not modeled
    - Max. likelihood point estimate only

### Goals
* Quantify uncertainty associated with deep network predictions
* Stochastic artificial neural networks trained using Bayesian methods
* Provide a mathematical framework to understand many regularization techniques and learning strategies that are already used in "classic" deep learning
* Enable the analysis of learning methods
* BNNs are also useful in online learning, where the model is retrained multiple times as new data become
available

### Bayesian Basics
* Probability is a measure of belief in the occurence of events
* Prior beliefs influence posterior beliefs in a reasoned way

### Bayesian Neural Networks
* Stochastic Neural Networks
    - Any of stochastic {activation, weights}
    - Simulates multiple possible models
    - Obtain a better idea of the uncertainty associated with the underlying processes
    - compare the predictions of multiple sampled model parametrizations $\theta$
        * If the different models agree, then the uncertainty is low. 
        * If they disagree, then the uncertainty is high
* Bayesian Neural Network Components
    - The model parametrization can be considered to be the hypothesis $H$ and the training set is the data $D$
    - Function Model: NN architecture
        * almost any model used for point estimate networks can be used as a functional model for a BNN
    - Stochastic Model
        * Equivalent to the choice of a loss function when training a point estimate neural network
        * Init: Prior distribution over the possible model parametrization
        * Probabilistic graphical models (PGMs)
            - use graphs to represent the interdependence of multivariate stochastic variables and subsequently decompose their probability distributions
            - Bayesian belief networks (BBN): directed acyclic graphs (DAGs)
                * Parents are sampled before their children. This is always possible since the graph is acyclic.
                * Observed variables are treated as the data. 
                * Unobserved, also called latent variables are treated as the hypothesis.
                * FIXME: IV.B, Page 5
    - Priors are often considered as soft constraints
    - Bayesian posterior for complex models such as artificial neural networks is a high dimensional and highly nonconvex probability distribution
* Prediction
    - The Bayesian posterior for complex models such as artificial neural networks is a high dimensional and highly nonconvex probability distribution
    - The probability distribution $p( y \mid x, D )$, called the marginal and which quantifies the model's uncertainty on its prediction, Sampled indirectly
        * Every ANN has an implicit prior prior explicit; BNNs, when used correctly, will at least make the prior  explicit
* Regression
    - Summarize the predictions of a BNN by model averaging. This approach is so common in ensemble learning that it is sometimes called "ensembling".
* Classification
    - Average model prediction will give the relative probability of each class, which can be  considered a measure of uncertainty
    - The final prediction is taken as the most likely class
    - This definition considers BNNs as discriminative models, i.e., models that aim to reconstruct a target variable $y$ given observations $x$. This excludes generative models
* Uncertainty
    - BNN allows distinguishing between the epistemic uncertainty (model uncert: $p( \theta \mid D )$) and the aleatoric uncertainty (class uncert: $p( y \mid x, \theta )$)
    - At prediction time, out-of-training distribution points will have high epistemic uncertainty instead of blindly giving a wrong prediction.


