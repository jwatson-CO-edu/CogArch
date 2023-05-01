# Hands-on Bayesian Neural Networks – A Tutorial for Deep Learning Users  
1. https://arxiv.org/pdf/2007.06823.pdf
1. https://github.com/french-paragon/BayesianNeuralNetwork-Tutorial-Metarepos 

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
            - The link between regularization and priors: Regularization acts as a soft constraint on the search space, in a manner similar to what a prior does for a posterior
            - Prior With a Consistency Condition $C(\mathbf{\theta},\mathbf{x})$ which is a function used to measure how well the model respects some hypothesis given a parametrization $\mathbf{\theta}$ and an input $\mathbf{x}$
    - Degree of Supervision and Alternative Forms of Prior Knowledge
        * Noisy Labels and Semi-Supervised Learning
            - In the case of noisy labels, one should extend the BBN to add a new variable for the noisy labels $\mathbf{\hat{y}}$ conditioned on $\mathbf{y}$
            - It is common, as the noise level itself is often unknown, to add a variable $\sigma$ to characterize the noise.
            - Data-driven regularization implies modifying the prior assumptions, and thus the stochastic model, to be able to extract meaningful information from the unlabeled dataset U.

* Markov Chain Monte Carlo (MCMC)
    - Construct a Markov chain, a sequence of random samples, which probabilistically depend only on the previous sample
    - Unlike standard sampling methods such as rejection or inversion sampling, most MCMC algorithms require an initial burn-in time before the Markov chain converges to the desired distribution.
    - The final collection of samples has to be stored after training, which is expensive for most deep learning models.
    - The most relevant MCMC method for BNNs is the Metropolis-Hastings algorithm (Algo 4). 
        * It does not require knowledge about the exact probability distribution to sample from. Instead, a function that is proportional to that distribution is sufficient.
        1. starts with a random initial guess
        1. samples a new candidate point around the previous using a proposal distribution
        1. If the new point is more likely than the previous according to the target distribution, it is accepted. If it is less likely, it is accepted with a certain probability or rejected otherwise.
    - Hamiltonian Monte Carlo algorithm (HMC)
        * Metropolis-Hasting algorithm for continuous distributions
        * HMC’s burn-in time is extremely short compared to the standard Metropolis-Hasting algorithm.

* Bayesian Inference Algorithms
    - A priori, a BNN does not require a learning phase as one just needs to sample the posterior and do model averaging (Algo 1)
    - For nontrivial models, even if the evidence has been computed, directly sampling the posterior is prohibitively difficult due to the high dimensionality of the sampling space.
    - MCMC algorithms are the best tools for sampling from the exact posterior.  However, their lack of scalability has made them less popular for BNNs
    - Variational Inference
        * scales better than MCMC algorithms
        * not an exact method
        * The idea is to have a distribution, called the variational distribution, parametrized by a set of parameters, rather than sampling from the exact posterior
        * The values of the parameters are then learned such that the variational distribution is as close as possible to the exact posterior.
        * The measure of closeness that is commonly used is the Kullback-Leibler divergence (KL-divergence)
            - It measures the differences between probability distributions based on Shannon’s information theory
            - The KL-divergence represents the average number of additional bits required to encode a sample from P using a code optimized for q
        * Stochastic variational inference (SVI), which is  the stochastic gradient descent method applied to variational inference
    - Bayes by Backpropagation
        * The main problem is that stochasticity stops backpropagation from functioning at the internal nodes of a network
        * Bayes-by-backprop is indeed a practical implementation of SVI combined with a reparametrization trick to ensure backpropagation works as usual.
        * The idea is to use a random variable as a nonvariational source of noise
        * A point is not sampled directly but obtained via a deterministic transformation
        * All other transformations being non-stochastic, backpropagation works as usual for the variational parameters
    - Inference Algorithms Adapted for Deep Learning
        * Being only approximately Bayesian is sufficient to achieve a correctly calibrated model with uncertainty estimates
        * Monte Carlo Dropout applied at evaluation time, is in fact variational inference with a variational distribution defined for each weight matrix
        * When used to train a BNN, dropout should not be seen as a regularization method, as it is part of the variational posterior, not the prior.This means that it should be coupled with a different type of regularization
    - Bayes via Stochastic Gradient Descent
        * The initial goal of SGD is to provide an algorithm that converges to an optimal point estimate solution while having only noisy estimates of the gradient of the objective function
        * To approximately sample the posterior using the SGD algorithm, a specific MCMC method, called stochastic gradient Langevin dynamic (SGLD) has been developed, Algo 7. SGLD offers better theoretical guarantees compared to other MCMC methods when the dataset is split into mini-batches.
    - Variational Inference Based on SGD Dynamic
        * SGD dynamic can be used as a variational inference method to learn a distribution by using Laplace approximation. Laplace approximation fits a Gaussian posterior by using the maximum a posteriori estimate as the mean and the inverse of the Hessian H of the loss (assuming the loss is the log likelihood) as covariance matrix

* Simplifying Bayesian Neural Networks
    - After training a BNN, one has to use Monte Carlo at evaluation time to estimate uncertainty. This is a major drawback of BNN
    - Bayesian Inference on the (N-)Last Layer(s) Only
        * aims to use only a few stochastic layers, usually positioned at the end of the networks
        * With only a few stochastic layers, training and evaluation can be drastically sped up while still obtaining meaningful results from a Bayesian perspective
    - Bayesian Teachers
        * The approach is to train a non-stochastic ANN to predict the marginal probability using a BNN as a teacher
        * This is related to the idea of knowledge distillation: possibly several pre-trained knowledge sources can be used to train a more functional system.



* Meta-learning, in the broadest sense, is the use of machine learning algorithms to assist in the training and optimization of other machine learning models
* Transfer learning designates methods that reuse some intermediate knowledge acquired on a given problem to address a different problem.
* Self-supervised learning is a learning strategy where the data themselves provide the "labels". Since the labels directly obtainable from the data do not match the task of interest, the problem is approached by learning a pretext (or proxy) task in addition to the task of interest.

