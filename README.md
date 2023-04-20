# CogArch
Experiments in AI, ML, and Cognitive Architecture. Mostly in C++.  
For educational purposes. This is to satisfy my own curiousity and deepen my understanding.  
The content herein is not guaranteed to be correct, functional, or readable. No warrantee offered or implied.  

# `DEV PLAN`

## Artificial Neural Networks
1. `[Y]` Restricted Boltzmann Machine, 2023-04: Re-write in C++ / Eigen3, 2023-04-06: Rewrite COMPLETED
    * I have lost trust in my Dlang implementation
    * Eigen does matx allocation and matx mult for me
    * `[Y]` Test reconstruction accuracy for full input, 2023-04-06: 0.819597, Reconstruction accuracy is 0.8% better than the D implementation
    * `[Y]` Test reconstruction accuracy for partial input, 2023-04-06: Turns out the test set was already partial
    * `[Y]` Add `typedef`s, 2023-04-08: Aliases work and save space

1. `[>]` Perceptrons + MLP, 2023-04: Re-write in C++ / Eigen3
    * `[Y]` Perceptron Margin Update, 2023-04-07: 0.99 accuracy
    * `[Y]` Add `typedef`s, 2023-04-08: Aliases work and save space
    * `[>]` Multi-Layer Perceptron + Backprop + MNIST Dataset
        - `[Y]` Issue: Apparent hanging and massive resource usage, 2023-04-09 - Sol'n: Datafile size read with wrong endianness and program was attemting to read 1.6B images!
            * `[Y]` Verify that next image is fetched, 2023-04-09: Image is clear and label matches!
            * `[Y]` Verify that next label is fetched, 2023-04-09: Image is clear and label matches!
        - `[>]` Issue: Rotten Performance
            * `[N]` If training examples are not shuffled, then shuffle them, 2023-04-13: Only 1 in ten examples is from a consecutive class
            * `[Y]` If perf still bad, then add trainable params, 2023-04-14: Network is bigger, but trains much more slowly (timewise)
            * `[Y]` If perf still bad, then do batch updates, 2023-04-14: Batches implemented, Learning still stalls at every learning rate and L1 rate attempted 
            * `[Y]` If perf still bad, then double-check gradient math, 2023-04-20: Consider this on good footing until further notice
                - Resources:
                    * `[Y]` [From scratch in C++, 1](https://www.youtube.com/watch?v=LA4I3cWkp1E), 2023-04-20: I have some doubts about their calcs of losses to pass to the prev layer
                        - Backprop @ 40:00
                    * `[Y]` [Backprop from Scratch](https://www.youtube.com/watch?v=Z97XGNUUx9o), 2023-04-20: Seems to corroborate the above
                    * [Live coding in C++](https://www.youtube.com/playlist?list=PL3MCKCM5GS4UmZJs-h6Rm11dlExhBPS-k)
                    * [From scratch in C++, 2 (by neuron, not by layer)](https://www.youtube.com/watch?v=sK9AbJ4P8ao)
                - `[ ]` Try a binary classifier to see if the network can identify one digit!
            * `[ ]` If perf still bad, then use gradient clipping?
            * `[ ]` If perf still bad, then use [Adam Optimizer](https://optimization.cbe.cornell.edu/index.php?title=Adam#:~:text=Adam%20optimizer%20is%20the%20extended,was%20first%20introduced%20in%202014.)

1. `[>]` Bayesian Neural Network (BNN, C++)
    * `[>]` Take notes and develop a dev plan
        - `[>]` https://arxiv.org/pdf/2007.06823.pdf
        - `[ ]` https://neptune.ai/blog/bayesian-neural-networks-with-jax
        - `[ ]` https://wjmaddox.github.io/assets/BNN_tutorial_CILVR.pdf
        - `[ ]` https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd
        - `[ ]` https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/variational_inference/bayesian_neural_network_advi.html
        - `[ ]` https://www.uv.es/gonmagar/blog/2018/03/15/BayesianNeuralNetworks
1. `[20]` Self-Organizing Map (SOM, C++)
    * `[ ]` Take notes and develop a dev plan
        - `[ ]` https://medium.com/machine-learning-researcher/self-organizing-map-som-c296561e2117
        - `[ ]` https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4
        - `[ ]` https://en.wikipedia.org/wiki/Self-organizing_map
        - `[ ]` http://www.pitt.edu/~is2470pb/Spring05/FinalProjects/Group1a/tutorial/som.html
        - `[ ]` http://blog.yhat.com/posts/self-organizing-maps-2.html
        - `[ ]` http://www.ai-junkie.com/ann/som/som1.html
        - `[ ]` https://github.com/abhinavralhan/kohonen-maps
1. `[50]` Graph Convolutional Network (GATv2, Python, Tensorflow)
    * `[ ]` Take notes and develop a dev plan
1. `[60]` Graph Generation with GCPN: Graph Convolutional Policy Network (You et al. 2018, Python, Tensorflow)  
    * `[ ]` Take notes and develop a dev plan
        - `[ ]` https://jian-tang.com/files/AAAI19/aaai-grltutorial-part3-generation.pdf
1. `[600]` Alternatives to Backpropagation (C++)
    * `[ ]` Take notes
        - `[ ]` Manneschi, Luca, and Eleni Vasilaki. "An alternative to backpropagation through time." Nature Machine Intelligence 2, no. 3 (2020): 155-156.
        - `[ ]` Bellec, Guillaume, Franz Scherr, Elias Hajek, Darjan Salaj, Robert Legenstein, and Wolfgang Maass. "Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets." arXiv preprint arXiv:1901.09049 (2019).
    * `[ ] DEV PLAN`, &nbsp;&nbsp; E: Which alternative to implement and why?



## The CORVID Project: Cognitive Architecture  
**C**ognition, **O**bservation, and **R**eaction for **V**aried **I**ntelligence **D**omains
1. `[31]` Evolutionary Feature Bus (EFB) 
    * `[ ]` T: Predict sine wave with +pi/2.0 phase shift
1. `[32]` Continuous, Interpolated Q-Learning (Julia)
    * `[ ]` Complete `DEV PLAN` as previously formulated
    * `[?]` Port highest-performing agent to C++?
1. `[33]` Active Inference Tutorial
    * `[34]` Think about how active inference is pertinent to the Partially Observable Lunar Lander.  Are there different behavior modes that are required?
1. `[40]` Run a POMDP solver on Partially Observable Lunar Lander
    * `[ ]` E: Which language to use?
    * `[ ]` POMCPOW or VOMCPOW
    * `[ ]` Can you add a learned transition model to it?
1. `[70]` Dynamic, Resizable ANN
    * `[ ]` E: Python lib(s)?
    * `[ ]` E: C++ lib(s)?
1. `[80]` CPPN-NEAT Architecture Search
    * `[ ]` E: Find most accessible existing implementation & test
1. `[1000]` CORVID Unit
    * `[ ]` Reconcile Unit daydreams with Active Inference
    * Recognizer
    * Learner / Agent
    * Amplifier
1. `[2000]` CORVID Stack / Agency, One stack
    * with input EFB
    * with output EFB
1. `[3000]` CORVID Cognitive Architecture
    * 2023-02-22 Brainstorming, Pt.1
        - Serial 
        - Recurrent Serial
        - Serial Medula
    * 2023-02-22 Brainstorming, Pt.2
        - Stack Sandwich
        - Parrallel Stack
    * Embedded domain mini-planner (Q-Learner?)

## Backend
* `[~~~]` Parallel Programming
    * If using D
        - https://dlang.org/phobos/std_parallelism.html
        - https://dlang.org/phobos/std_process.html
    * If using C++
        - pthread
        - MPI
        - OpenCL
* `[~~~]` [Multi-Layer Perceptron + OpenCL?](https://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl)  

## Physics
1. `[~~~]` [Geometric Algebra for Computer Science]
    * 2023-04-09: No D implementation  
        - (https://www.amazon.com/Geometric-Algebra-Computer-Science-Revised/dp/0123749425) @ Gemmill Library
        - https://rosettacode.org/wiki/Geometric_algebra
    * C++ 
        - https://github.com/laffernandes/gatl
        - http://versor.mat.ucsb.edu/
        - https://bivector.net/lib.html


## Frameworks & Environments
#### YOU SHOULD AVOID BUILDING AN ENVIRONMENT IF A SIMILAR ONE IS FREELY AVAILABLE!
1. `[~~~~]` Braitenberg 2D Grid World
    * Agents situated at `<X,Y,D>` where `<X,Y>` is a cell address and `<D>` is a cardinal direction
    * Components have no extent, but possibly oriented in a cardinal direction
    * `[ ]` Add EFB
    * `[ ]` Add Continuous Q-Learning
1. `[~~~~]` Braitenberg 3D Grid World
1. `[~~~~]` Continous 2D Braitenberg World
1. `[~~~~]` Continous 3D Braitenberg World

## Notes
* `[2023-03-28]` Switching to [C++](https://en.cppreference.com/w/) from [D](https://dlang.org/) to mentally prepare for job search. 80% of surveyed positions mention C++; None mention D.
    - I also caused myself some confusion with  assignment, slicing, and equivalence of dynamic arrays
    - If I pick D back up:
        * Evaluate recent language changes
        * Evaluate state of needed libraries
            - I need a satisfying linear algebra library
            - Raylib
# Resources
* [Top 14 C++ Machine Learning Libraries](https://anywhere.epam.com/business/c-plus-plus-ml-libraries)
    - FANN?
* [Adam Optimizer](https://optimization.cbe.cornell.edu/index.php?title=Adam#:~:text=Adam%20optimizer%20is%20the%20extended,was%20first%20introduced%20in%202014.)
