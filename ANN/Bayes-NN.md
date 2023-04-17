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
    - Simulates multiple possible models: Can be considered a special case of Ensemble Learning
        * We get the benefit of aggregating the results from a collection of mediocre models
        * We **also** get the benefit of knowing the uncertainty / spread of the results
    - Obtain a better idea of the uncertainty associated with the underlying processes
    - compare the predictions of multiple sampled model parametrizations $\theta$
        * If the different models agree, then the uncertainty is low. 
        * If they disagree, then the uncertainty is high
* Bayesian Neural Network Components
    - The model parametrization can be considered to be the hypothesis $H$ and the training set is the data $D$
    - Requires that we have a prior distribution, but this is not a problem because not only can we quantify its uncertainty, we can also update this distribution
    - Functional Model: NN architecture
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
* Active Learning
    - Data points in the training set with high epistemic uncertainty are scheduled to be labeled with higher priority
* Uncertainty
    - BNN allows distinguishing between the epistemic uncertainty (model uncert: $p( \theta \mid D )$) and the aleatoric uncertainty (class uncert: $p( y \mid x, \theta )$)
    - At prediction time, out-of-training distribution points will have high epistemic uncertainty instead of blindly giving a wrong prediction.

* Bayesian Neural Network Design
    - Function Model: almost any model used for point estimate networks can be used as a functional model for a BNN
        * S. Pouyanfar, S. Sadiq, Y. Yan, H. Tian, Y. Tao, M. P. Reyes, M.-L. Shyu, S.-C. Chen, and S. S. Iyengar, “A survey on deep learning: Algorithms, techniques, and applications,” ACM Comput. Surv., vol. 51, no. 5, Sep. 2018.
    - Stochastic Model
        * Probabilistic Graphical Model (PGM) --to-> BNN Stochastic Model
            - Setting the priors
                * For basic architectures such as Bayesian regression, a standard procedure is to use a normal prior with a zero mean and a diagonal covariance.  This is equivalent to an L2 normalization
            - DNN have many equivalent parameterizations, they are overparameterized: Statistical unidentifiability
                * This can lead to complex multimodal posteriors
                * Weight Space Symmetry: One can build an equivalent parametrization of an ANN with at least one hidden layer. 
                    - This is achieved by permuting two rows in $\mathbf{\underline{W}_i}$, as the corresponding columns in the following layer’s weight matrix $\mathbf{\underline{W}_{i+1}}$
                    - as the number of hidden layers and the number of units in the hidden layers grow, the number of equivalent representations, which would roughly correspond to the modes in the posterior distribution, grows factorially
                    - Mitigation strategy: enforce the bias vector in each layer to be sorted in an ascending or a descending order.
                    - weightspace symmetry may implicitly support the exploration of the parameter space during the early stages of the optimization.
                * Scaling Symmetry: arises when using nonlinearities with the property $s(\alpha x) = \alpha s(x)$, which is the case of RELU and Leaky-RELU
                    - Scaling symmetry influences the posterior shape, making it harder to approximate
                    - Using a Gaussian prior reduces the scaling symmetry problem, as it favors weights with the same Frobenius norm on each layer
                    - A soft version of the activation normalization can also be implemented by using a consistency condition
                    - Givens transformations (also called Givens rotations) have been proposed as a mean to constrain the norm of the hidden layers and address the scaling symmetry issue
            - The link between regularization and priors
                * FIXME: pg 6, col 2, bottom


