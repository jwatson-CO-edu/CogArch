# FANN Install
1. `cd /tmp`
1. `git clone https://github.com/libfann/fann.git`
1. `cd fann`
1. `cmake .`
1. `sudo make install`
1. `sudo ldconfig`
# PyTorch
1. `python3.9 -m pip install --upgrade pip`
1. `python3.9 -m pip install --upgrade jupyterlab --user`
1. `python3.9 -m pip install git+https://github.com/pyg-team/pytorch_geometric.git --user`
# Graph-NN Overview
https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
* Graph Neural Networks are special types of neural networks capable of working with a graph data structure. They utilize CNNs and graph embeddings
    - The input graph structure is converted into graph embedding, allowing us to maintain information on nodes, edges, and global context. 
* Types of Graph Neural Networks
    - **Graph Convolutional Networks (GCNs)**: 
        * Similar to traditional CNNs. It learns features by inspecting neighboring nodes. GNNs aggregate node vectors, pass the result to the dense layer, and apply non-linearity using the activation function. 
        * It consists of Graph convolution, linear layer, and non-learner activation function. 
        * There are two major types of GCNs: 
            - **Spatial Graph Convolutional Networks**: use spatial features to learn from graphs that are located in spatial space.  
            - **Spectral Graph Convolutional Networks** use Eigen-decomposition of graph Laplacian matrix for information propagation along nodes. 
    - **Graph Auto-Encoder Networks**: 
        * Learn graph representation using an encoder and attempt to reconstruct input graphs using a decoder. 
        * The encoder and decoders are joined by a bottleneck layer. 
        * They are commonly used in link prediction as Auto-Encoders are good at dealing with class balance. 
    * **Recurrent Graph Neural Networks (RGNNs)**: 
        - Can handle multi-relational graphs where a single node has multiple relations. 
        - This type of graph neural network uses regularizers to boost smoothness and eliminate over-parameterization. 
        - Used in generating text, machine translation, speech recognition, generating image descriptions, video tagging, and text summarization.
    * **Gated Graph Neural Networks (GGNNs)** 
        - Better than the RGNNs in performing tasks with long-term dependencies. 
        - Gated Graph Neural Networks improve Recurrent Graph Neural Networks by adding a node, edge, and time gates on long-term dependencies. 
        - Similar to Gated Recurrent Units (GRUs), the gates are used to remember and forget information in different states. 
* Types of Graph Neural Networks Tasks
    - **Graph Classification**: we use this to classify graphs into various categories. Its applications are social network analysis and text classification. 
        * Predict whether a subnet is suitable to a purpose?
    - **Node Classification**: this task uses neighboring node labels to predict missing node labels in a graph. 
        * Classify the type of object that the node represents
    - **Link Prediction**: predicts the link between a pair of nodes in a graph with an incomplete adjacency matrix. It is commonly used for social networks. 
        * Predict whether similar objects imply an analogous connection
    - **Community Detection**: divides nodes into various clusters based on edge structure. It learns from edge weights, and distance and graph objects similarly. 
        * Predict whether a collection of engrams should mutually activate
    - **Graph Embedding**: maps graphs into vectors, preserving the relevant information on nodes, edges, and structure.
Graph Generation: learns from sample graph distribution to generate a new but similar graph structure. 
# Graph Convolutional Network (GCN)
* The convolution in GCN is the same as a convolution in image CNNs. It multiplies neurons with filters to learn from data features.
* A model learns the features from neighboring nodes.
# Recurrent Graph Neural Networks (RGNNs)
# Gated Graph Neural Networks (GGNNs)