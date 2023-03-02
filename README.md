# CogArch
Experiments in AI, ML, and Cognitive Architecture. Mostly in [D](https://dlang.org/).  
For educational purposes. This is to satisfy my own curiousity and deepen my understanding.  
The content herein is not guaranteed to be correct, functional, or readable. No warrantee offered or implied.  

# `DEV PLAN`

## Artificial Neural Networks
1. `[Y]` Restricted Boltzmann Machine
1. `[>]` Perceptrons
    * `[Y]` Perceptron Margin Update    
    * `[>]` Multi-Layer Perceptron + Backprop + MNIST Dataset
1. `[10]` Bayesian Neural Network (BNN)
1. `[20]` Self-Organizing Map (SOM)
1. `[50]` Graph Convolutional Network (GATv2)
1. `[60]` Graph Generation with GCPN: Graph Convolutional Policy Network ( You et al. 2018)  
    * https://jian-tang.com/files/AAAI19/aaai-grltutorial-part3-generation.pdf
1. `[600]` Alternatives to Backpropagation
    * `[ ]` Read: Manneschi, Luca, and Eleni Vasilaki. "An alternative to backpropagation through time." Nature Machine Intelligence 2, no. 3 (2020): 155-156.
    * `[ ]` Read: Bellec, Guillaume, Franz Scherr, Elias Hajek, Darjan Salaj, Robert Legenstein, and Wolfgang Maass. "Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets." arXiv preprint arXiv:1901.09049 (2019).
    * `[ ]` E: Which alternative to implement and why?
    * `[ ]` Implement!



## The CORVID Project: Cognitive Architecture  
**C**ognition, **O**bservation, and **R**eaction for **V**aried **I**ntelligence **D**omains
1. `[31]` Evolutionary Feature Bus (EFB) 
    * `[ ]` T: Predict sine wave with +pi/2.0 phase shift
1. `[32]` Continuous, Interpolated Q-Learning
    * `[ ]` Complete `DEV PLAN` as previously formulated
    * `[?]` Port highest-performing agent to [D](https://dlang.org/)?
1. `[33]` Active Inference Tutorial
    * `[34]` Think about how active inference is pertinent to the Partially Observable Lunar Lander.  Are there different behavior modes that are required?
1. `[40]` Run a POMDP solver on Partially Observable Lunar Lander
    * POMCPOW or Voronoi Equivalent
    * Can you add a learned transition model to it?
1. `[70]` Braitenberg 2D Grid World
    * Agents situated at `<X,Y,D>` where `<X,Y>` is a cell address and `<D>` is a cardinal direction
    * Components have no extent, but possibly oriented in a cardinal direction
    * `[ ]` Add EFB
    * `[ ]` Add Continuous Q-Learning
1. `[80]` Dynamic, Resizable ANN
    * Does [VectorFlow](https://netflixtechblog.medium.com/introducing-vectorflow-fe10d7f126b8) have a dynamic option.  If not, can you extend the ANN layer class to allow this?
1. `[90]` CPPN-NEAT Architecture Search
1. `[100]` Braitenberg 3D Grid World
1. `[300]` Continous 2D Braitenberg World
1. `[500]` Continous 3D Braitenberg World

## Backend
* `[700]` Parallel Programming
    - https://dlang.org/phobos/std_parallelism.html
    - https://dlang.org/phobos/std_process.html
* `[800]` [Multi-Layer Perceptron + OpenCL?](https://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl)  

## Physics
1. `[900]` [Geometric Algebra for Computer Science](https://www.amazon.com/Geometric-Algebra-Computer-Science-Revised/dp/0123749425) @ Gemmill Library


