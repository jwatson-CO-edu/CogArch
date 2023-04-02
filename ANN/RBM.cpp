/*
RBM.cpp
James Watson, 2023-04
Simplest Restricted Boltzmann Machine (RBM) example
From:
* https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
* https://www.youtube.com/watch?v=wMb7cads0go
* https://www.youtube.com/watch?v=S0kFFiHzR8M
* https://www.youtube.com/watch?v=n26NdEtma8U
* https://www.youtube.com/watch?v=iPuqoQih9xk
* Hinton, Geoffrey E. "A practical guide to training restricted Boltzmann machines." 
  Neural Networks: Tricks of the Trade: Second Edition (2012): 599-619.

Compile & Run:
rdmd RBM.d


///// Background /////

Boltzmann Machine is a generative unsupervised model, which involves learning a probability distribution 
from an original dataset and using it to make inferences about never before seen data.

Boltzmann Machine has an input layer (also referred to as the visible layer) and one or several hidden layers.
Neurons generate information regardless they are hidden or visible.

RBM’s objective is to find the joint probability distribution that maximizes the log-likelihood function.

Since RBMs are undirected, they don’t adjust their weights through gradient descent and backpropagation. 
They adjust their weights through a process called contrastive divergence. At the start of this process, 
weights for the visible nodes are randomly generated and used to generate the hidden nodes. 
These hidden nodes then use the same weights to reconstruct visible nodes. The weights used to 
reconstruct the visible nodes are the same throughout.

Now, the difference v(0)−v(1) can be considered as the reconstruction error that we need to reduce 
in subsequent steps of the training process. So the weights are adjusted in each iteration so as to 
minimize this error.

*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

/// Standard Imports ///
#include <cmath>// ---- `exp` 
#include <stdlib.h> //- `srand`, `rand` 
#include <time.h> // -- `time` 
#include <algorithm> // `clamp`

/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;



////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////

float randf(){
    // Return a pseudo-random number between 0.0 and 1.0
    return  1.0f * rand() / RAND_MAX;
}

float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }

float Box_Muller_normal_sample( float mu = 0.0f, float sigma = 1.0f ){
    // Transform 2 uniform samples into a zero-mean normal sample
    // Source: https://www.baeldung.com/cs/uniform-to-normal-distribution
    float u1 = randf();
    float u2 = randf();
    return mu + sqrt( -2.0 * log( u1 ) ) * cos( 2.0 * M_PI * u2 ) * sigma;
}

////////// RESTRICTED BOLTZMANN MACHINE ////////////////////////////////////////////////////////////
struct RBM{
    // Simplest Restricted Boltzmann Machine 

    uint /**/ dI; // Input  dimensions
    uint /**/ dH; // Hidden dimensions
    MatrixXd W; //- Weight matrix
    MatrixXd b; //- Hidden  unit biases
    MatrixXd c; //- Visible unit biases
    MatrixXd h; //- Hidden  unit values
    MatrixXd v; //- Visible unit values
    MatrixXd x; //- Input values
    float    lr; // Learning rate

    RBM( uint inputDim, uint hiddenDim, float learnRate ){
        // Allocate arrays and init all values to zero
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Set params
        dI = inputDim;
        dH = hiddenDim;
        lr = learnRate;
        
        // Init weights
        W = MatrixXd{ dI, dH };
        b = MatrixXd{ dH,  1 };
        h = MatrixXd{ dH,  1 };
        c = MatrixXd{ dI,  1 };
        v = MatrixXd{ dI,  1 };
        x = MatrixXd{ dI,  1 };
    }

};