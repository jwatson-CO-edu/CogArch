# CogArch
Experiments in AI, ML, and Cognitive Architecture. Mostly in C++.  
For educational purposes. This is to satisfy my own curiousity and deepen my understanding.  
The content herein is not guaranteed to be correct, functional, or readable. No warrantee offered or implied.  

# `DEV PLAN`

## The BLUE-J Project: Neurosymbolic Agent
**B**asic **L**anguage **U**nderstanding with **E**ngram-**J**umps  
* Knowledge/Concept graphs as the basic unit of cognition and memory
    - Should be probabilistic in nature
* Connections, Sequences, and Relationships represented by rich probabilistic Edges
* Relevant structures and edges should inspire and activate relevant agent actions
* Goals: 
    - Build and operate over structures that simulate the structure of the environment.
    - Build multimodal knowledge graphs with visual, state, and semantic data
    - Learn generalizations over grounded graphs that improve decision-making
    - Extend LLMs with engrams that allow optimal, long-term reasoning
* Daydreams: 
    - Conitive Graphs (Engrams) that can represent all data required for perception, action, and planning
        * Relationships
        * Memory
        * Objects, Percepts
        * Affordances, Options
        * Actions
        * States, State Changes, Sequences, Episodes
    - Personal Assistant
        * Digital
        * Physical
    - Enable exchange of rich knowledge between robots
### Preliminary: 
#### Graph Neural Networks
1. `[50]` Graph Convolutional Network (GATv2, Python, Tensorflow)
    * `[Y]` Take notes and develop a dev plan, 2023-11-20: Identified major types of GNN that *may* require further study
    * `[Y]` Choose and install library, 2023-11-19: FANN, `torch_geometric`, See installation instructions in "Graph-NN_01.md"

1. `[60]` Graph Generation with GCPN: Graph Convolutional Policy Network (You et al. 2018, Python, Tensorflow)  
    * `[ ]` Take notes and develop a dev plan
        - `[ ]` https://jian-tang.com/files/AAAI19/aaai-grltutorial-part3-generation.pdf

#### Decision Making Under Uncertainty: POMDP
1. `[ ]` Implement simplest MCTS
1. `[55]` Run a POMDP solver on Partially Observable Lunar Lander
    * `[ ]` E: Which language to use?
    * `[ ]` POMCPOW or VOMCPOW
    * `[ ]` Can you add a learned transition model to it?

### Stage 0: Engram-guided MCTS
* `[ ]` Q: What semantic data can be added to states / samples automatically?
* `[ ]` Implement semantically "decorated" state graphs
* `[ ]` Learn a value function over "decorated" state graphs
* `[ ]` Demonstrate improved performance when we preferentially sample from high-value state grapghs
### Stage 1: Planning over engrams with Hand-Coded Architecture
Concept: Replace pure symbols with probabilistic graphical symbols in both deliberative and reactive planning as a means to achieve both generality and adaptability.
#### Reactive?
Concept: BOBCAT over knowledge graphs
#### Deliberative?
Concept: TAMP over knowledge graphs
#### Reactive + Deliberative?
Concept: MAGPIE over knowledge graphs
### Stage 2: Neurosymbolic Plan Synthesis
Concept: Extend probabilistic graphical symbols to probabilistic graphical plans that contain rich representation of state changes and options.
### Stage 3: Neurosymbolic Probabilistic Knowledge Graph Architecture
Concept: Extend probabilistic knowledge graphs into a complete cognitive architecture that handles learning, recall, and planning in a cohesived and flexible manner.

### Exploratory: Neuro-Cognitive Elements

1. `[70]` Dynamic, Resizable ANN, Neuroplastic capability
    * `[ ]` E: Python lib(s)?
    * `[ ]` E: C++ lib(s)?
1. `[80]` CPPN-NEAT Architecture Search
    * `[ ]` E: Find most accessible existing implementation & test

## Artificial Neural Networks

1. `[600]` Alternatives to Backpropagation (C++)
    * `[ ]` Take notes
        - `[ ]` Manneschi, Luca, and Eleni Vasilaki. "An alternative to backpropagation through time." Nature Machine Intelligence 2, no. 3 (2020): 155-156.
        - `[ ]` Bellec, Guillaume, Franz Scherr, Elias Hajek, Darjan Salaj, Robert Legenstein, and Wolfgang Maass. "Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets." arXiv preprint arXiv:1901.09049 (2019).
    * `[ ] DEV PLAN`, &nbsp;&nbsp; E: Which alternative to implement and why?



## The CORVID Project: Cognitive Architecture  
**C**ognition, **O**bservation, and **R**eaction for **V**aried **I**ntelligence **D**omains

### Deep Reinforcement Learning

1. `[32.10]` Hugging-Face Deep-RL MOOC

&nbsp;

1. `[33]` Active Inference Tutorial
    * `[34]` Think about how active inference is pertinent to the Partially Observable Lunar Lander.  Are there different behavior modes that are required?

### Exploratory: Neuro-Cognitive Elements

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

### Applications and Tests

1. `[4000]` HOUSE-Bot: Household Online Understanding System for Execution
    * `[4100]` Navigate rooms with robust SLAM
        - `[ ]` Build/buy a robust but reasonable platform
            * `[ ]` Add perception
            * `[ ]` Add data logging
        - `[ ]` NAMO: Navigation Among Movable Obstacles
    * `[4200]` Mobile Manipulator
        - `[ ]` Build/buy a robust but reasonable manipulator
            * `[ ]` Add perception
        - `[ ]` Integrate platform + manipulator
        - `[ ]` TAMP: Task and Motion Planning
            * `[ ]` Implement
            * `[ ]` Extend
    * `[4300]` Lifelong Learning
        - Increase capability one task and concept at a time
            * Sweep
            * Vacuum? Dust?
            * Identify, Pick up, and discard clutter
            * Identify, Pick up, and put away objects
            * ?? More ??
        - Read intelligence "philosophy"
        - Choose simplest and most applicable cognitive architecture
            * ONLY add complexity as NEEDED
        - Re-implement TAMP in the context of this architecture
    * `[4400]` Stretch Goals
        - Training by example
        - Training from verbal input

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

1. `[~~~]` Modeling Plasma Physics  
    1. `[ ]` Find a small scale plasma experiment online
    1. `[ ]` Repeat experiment
    1. `[ ]` Find a plasma model in literature
    1. `[ ]` Model experiment with Geometric Algebra (See above)
    1. `[ ]` Model stellerator
    1. `[ ]` Model "Inductive Fusion"
    * `{ }` Are quantum mechanics required?
    * `{ }` Are relativistic effects considered?

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


# Completed Projects

1. `[Y]` Evolutionary Feature Bus (EFB), 2023-08-09: Fitness continually improves, at this point I don't particularly care how well, but there are improvements to be made if I resume the project 
    * `[Y]` T: Output with sine input, 2023-06-30: Works as designed
    * `[Y]` Random feature generation, 2023-08-09: Random operations with random input features
    * `[Y]` Average fitness evaluation after N steps, 2023-08-09: Culling and reporduction works
    * `[Y]` Cull and repopulate according to fitness, 2023-08-09: Culling and reporduction works
    * `[Y]` T: Predict sine wave with +pi/2.0 phase shift, 2023-08-09: Fitness continually improves, at this point I don't particularly care how well, but there are improvements to be made if I resume the project
        - Plot outputs for evaluation and troubleshooting
        - Find out why output calculation is so slow and fix it
        - Test as input to a neural network and make necessary changes
        - Test as output to a neural network and make necessary changes
        - Test as an intermediate layer between neural network layers and make necessary changes
    * `{N}` Parameter backprop?, 2023-08-09: This will not be persued at this time.  It's not fun right now.

1. `[Y]` Self-Organizing Map (SOM, C++), 2023-06-19: Seems to converge, not going to count clusters at this time
    ```
    ########## Self-Organizing Map Report ##########
    Average distance to BMU: 171.483
    Final Search Radius: ___ 633.237
    Learning Rate: _________ 0.02
    Problem Scale: _________ 800
    Decay Constant: ________ 0.9999
    ```
    * `[Y]` Take notes and develop a dev plan, 2023-05-09: Code might be useful
        - `[Y]` https://medium.com/machine-learning-researcher/self-organizing-map-som-c296561e2117, 2023-05-09
        - `[Y]` https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4, 2023-05-09: This is just the wikipedia alticle
        - `[Y]` https://en.wikipedia.org/wiki/Self-organizing_map, 2023-05-09
        - `[N]` http://www.pitt.edu/~is2470pb/Spring05/FinalProjects/Group1a/tutorial/som.html, 2023-05-09
        - `[~]` http://www.ai-junkie.com/ann/som/som1.html, 2023-05-09: Code might be useful
        - `[~]` https://github.com/abhinavralhan/kohonen-maps, 2023-05-09: Code might be useful
    * `DEV PLAN`
        - `[Y]` Choose a classification/inference problem (from the reading plan if available), 2023-06-05: Seizure Datasets
1. `[Y]` Perceptrons + MLP, 2023-04: Re-write in C++ / Eigen3, 2023-05-18: 80% accuracy achieved, anything above this is just tuning.  This is pretty poor performance, but in the case that MLP will be applied to a C++ project, I will use an established library for it.
    Running Best:  

    ```
    Validation Accuracy: 0.8157
    Trained for 117 epochs!

    ### Network Summary ###
    Learning Rate: 3e-05
    Layer 1: Input 784  X  Output 64
    Layer 2: Input 64  X  Output 32
    Layer 3: Input 32  X  Output 10
    Each layer's gradient scaled to 10
    Mini-batches of size 0!
    Data shuffled each epoch!
    # End of Summary #
    ```
    * `[Y]` Perceptron Margin Update, 2023-04-07: 0.99 accuracy
    * `[Y]` Add `typedef`s, 2023-04-08: Aliases work and save space
    * `[Y]` Multi-Layer Perceptron + Backprop + MNIST Dataset, 2023-05-18: 80% accuracy achieved, anything above this is just tuning
        - `[Y]` Issue: Apparent hanging and massive resource usage, 2023-04-09 - Sol'n: Datafile size read with wrong endianness and program was attemting to read 1.6B images!
            * `[Y]` Verify that next image is fetched, 2023-04-09: Image is clear and label matches!
            * `[Y]` Verify that next label is fetched, 2023-04-09: Image is clear and label matches!
        - `[Y]` Issue: Rotten Performance
            * `[N]` If training examples are not shuffled, then shuffle them, 2023-04-13: Only 1 in ten examples is from a consecutive class
            * `[Y]` If perf still bad, then add trainable params, 2023-04-14: Network is bigger, but trains much more slowly (timewise)
            * `[Y]` If perf still bad, then do batch updates, 2023-04-14: Batches implemented, Learning still stalls at every learning rate and L1 rate attempted 
            * 2023-05-03: Found a potential problem that might have been causing the biases not to update, steadiliy creeping above random performance
            * `[Y]` If perf still bad, then double-check gradient math, 2023-04-20: Consider this on good footing until further notice, 2023-05-15: Grad descent and backprop are working. 0.65 accuracy, far above random.  Needs more tuning, try ADAM.
                - Resources:
                    * `[Y]` [Sebastian Lague Tut](https://www.youtube.com/watch?v=hfMk-kjRv4c), 2023-05-15: Not helpful
                    * `[Y]` [From scratch in C++, 1](https://www.youtube.com/watch?v=LA4I3cWkp1E), 2023-04-20: I have some doubts about their calcs of losses to pass to the prev layer
                        - Backprop @ 40:00
                    * `[Y]` [Backprop from Scratch](https://www.youtube.com/watch?v=Z97XGNUUx9o), 2023-04-20: Seems to corroborate the above
                    * [Live coding in C++](https://www.youtube.com/playlist?list=PL3MCKCM5GS4UmZJs-h6Rm11dlExhBPS-k)
                    * [From scratch in C++, 2 (by neuron, not by layer)](https://www.youtube.com/watch?v=sK9AbJ4P8ao)
            * `[Y]` If perf still bad, then use gradient clipping?, 2023-04-22: Gradient *scaling* results in *faster training* but **NOT** *better perf*
            * `[Y]` No really, actually **shuffle** the data **each epoch**, 2023-04-28: Shuffled, but found a promising error
                - `[N]` Try a binary classifier to see if the network can identify one digit!
            
            * `[N]` If perf still bad, then use [Adam Optimizer](https://optimization.cbe.cornell.edu/index.php?title=Adam#:~:text=Adam%20optimizer%20is%20the%20extended,was%20first%20introduced%20in%202014.), 2023-05-18: 80% accuracy achieved, anything above this is just tuning

1. `[Y]` Restricted Boltzmann Machine, 2023-04: Re-write in C++ / Eigen3, 2023-04-06: Rewrite COMPLETED
    * I have lost trust in my Dlang implementation
    * Eigen does matx allocation and matx mult for me
    * `[Y]` Test reconstruction accuracy for full input, 2023-04-06: 0.819597, Reconstruction accuracy is 0.8% better than the D implementation
    * `[Y]` Test reconstruction accuracy for partial input, 2023-04-06: Turns out the test set was already partial
    * `[Y]` Add `typedef`s, 2023-04-08: Aliases work and save space

# Suspended Projects

1. `[P]` Continuous, Interpolated Q-Learning (Julia), Complete `DEV PLAN` as previously formulated  
    **Stopping Criterion**: Averge of $\geq 30.0$ seconds vertical across 64 runs
    * `[ ]` Test with discrete Q-Learning FIRST
    
    * `[Y]` Implement Temporal Difference target, 2023-08-22: Seems overall better, but requires tuning
    * `[P]` TD Learning
        - `[P]` Tune learning rate
        - `[ ]` Tune swap frequency
    * `[ ]` Eligibility Traces
        - `[ ]` Backtracking method
        - `[ ]` Sutton & Barto Method
    * `[ ]` Actor-Critic
        - `[ ]` Separate the policy and the value function
        - `[ ]` Probabilistic action selection
        - `{?}` Dyna-AC
    * `{?}` Port highest-performing agent to C++?

&nbsp;

1. `[P]` Continuous, Interpolated Q-Learning (Julia), Complete `DEV PLAN` as previously formulated  
    **Stopping Criterion**: Averge of $\geq 30.0$ seconds vertical across 64 runs
    * `[Y]` Test structures, 2023-08-09: Everything seems to be in place
    * `[N]` Reinstate best settings from the report, 2023-08-22: The model from the report **was not properly tuned**!
    * `[Y]` Implement Temporal Difference target, 2023-08-22: Seems overall better, but requires tuning
    * `[P]` Discrete TD Learning
    * `[P]` TD Learning
        - `[P]` Tune learning rate
        - `[ ]` Tune swap frequency
    * `[ ]` Eligibility Traces
        - `[ ]` Backtracking method
        - `[ ]` Sutton & Barto Method
    * `[ ]` Actor-Critic
        - `[ ]` Separate the policy and the value function
        - `[ ]` Probabilistic action selection
        - `{?}` Dyna-AC
    * `{?}` Port highest-performing agent to C++?

&nbsp;

1. `[~]` Bayesian Neural Network (BNN, C++), 2023-06-05: The convolutional BNN is a complex network that would take significant time to make from scratch, Delaying this in favor of other projects
    * `[Y]` Reading Plan
        - `[Y]` https://arxiv.org/pdf/2007.06823.pdf, 2023-05-01
        - `[Y]` https://sanjaykthakur.com/2018/12/05/the-very-basics-of-bayesian-neural-networks/, 2023-05-04
        - `[Y]` https://neptune.ai/blog/bayesian-neural-networks-with-jax, 2023-05-05
        - `[~]` https://wjmaddox.github.io/assets/BNN_tutorial_CILVR.pdf, 2023-05-05
        - `[N]` https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd, 2023-05-05: Not so helpful
        - `[N]` https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/variational_inference/bayesian_neural_network_advi.html
        - `[N]` https://www.uv.es/gonmagar/blog/2018/03/15/BayesianNeuralNetworks, 2023-05-05: Maths and proofs if I need them
        - 
    * `DEV PLAN`
        - `[Y]` Choose (Distributiuon over Weights) -OR- (Distribution over Activations), 2023-05-19: Distribution over Activations will be simpler
        - 2023-05-23: Need more detail in order to write the implementation, Mini Reading Plan:
            * `[>]` Y. Gal, R. Islam, and Z. Ghahramani, “Deep Bayesian active learning with image data,” in Proc. 34th Int. Conf. Mach. Learn., 2017, vol. 70, pp. 1183 –1192.
            * `[ ]` J. Zeng, A. Lesnikowski, and J. M. Alvarez, “The relevance of Bayesian layer positioning to model uncertainty in deep Bayesian active learning,” 2018. [Online]. Available: http://ar xiv.org/abs/1811.12535
        - `[Y]` Choose a classification/inference problem, 2023-05-26: MNIST OK, Maybe skin cancer diagnosis from lesion images (ISIC2016 task)
            * Here?: https://github.com/french-paragon/BayesianNeuralNetwork-Tutorial-Metarepos 
        - `[ ]` Test normal sampling function from RBM, is it Guassian?
        - `[ ]` Basic BNN Class
        - `[ ]` Basic BNN Test

