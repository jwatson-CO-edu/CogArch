# Self-Organizing (Feature) Map &nbsp; -or- &nbsp; Kohonen Map
## Sources
* https://medium.com/machine-learning-researcher/self-organizing-map-som-c296561e2117

&nbsp;  

## Notes
* What and Why
    - Self-supervised
    - Competitive learning network
    - Clustering: 
        * discretized representation of the input space of the training samples
        * membership of the input data
    - Dimensionality reduction
* Drawbacks
    - It does not build a generative model 
    - It does not do well when using categorical data, even worse for mixed types data.

&nbsp;  

* Clustering (K-means)
    - partition $n$ observations into $k$ clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype (centroid) of the cluster.
    - Seed with centroids
    - Updates centroids 

&nbsp;  

* SOM Basics
    - Lattice of nodes fully connected to the input layer
        * The input layer has the same dimensionality of the input
        * The lattice has the same (topological) dimensionality of the input, though it may have thousands of nodes
        * Each has a specific topological condition
        * vector of weights of the same dimension as the input vectors
        * There are no lateral connections between nodes within the lattice.
        * The lattice nodes begin arbitralily distributed in the input space
        * The lattice should have molded itself to the data distribution by the end of training
    - Weights express the coordinates of the nodes
        - There is no activation function
    - Learning Process
        * where the node weights match the input vector, that area of the lattice is selectively optimized to more closely resemble the data for the class the input vector belongs to.
        * Goal: SOM eventually settles into a map of stable zones
        1. Each node’s weights are initialized.
        1. Loop Begin
        1. A vector is chosen at random from the set of training data and presented to the lattice.
        1. Every node is examined to calculate which ones weights are most like the input vector. The winning node is commonly known as the Best Matching Unit (BMU).
            * Typical to use closest Euclidean distance
        1. The radius of the neighborhood of the BMU is now calculated. Any nodes found within this radius are deemed to be inside the BMU’s neighborhood.
            * This is a value that starts large, typically set to the ‘radius’ of the lattice, but diminishes each time-step.
            * Membership is determined by an exponential decay function
            * In unsupervised classification, $\sigma$ is sometimes based on the Euclidean distance between the centroids of the first and second closest clusters.
        1. Each neighboring node’s  weights are adjusted to make them more like the input vector. 
            * The closer a node is to the BMU; the more its weights get altered.
        1. Loop for $N$ iterations

&nbsp;  

* [Alternatives and Extensions](https://en.wikipedia.org/wiki/Self-organizing_map#Alternatives)
    - time adaptive self-organizing map
    - growing self-organizing map 
    - elastic maps
    - Conformal matching / interpolation
    - oriented and scalable map