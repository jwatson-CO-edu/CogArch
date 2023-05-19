/*
MLP.cpp
James Watson, 2023-04
Simplest demo of a scratch-built multi-layer perceptron (MLP)
g++ MLP.cpp -std=gnu++17 -O3 -I /usr/include/eigen3 -o p1.out

WARNING: This implementation is absurdly unoptimized!
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

///// Imports ////////////////////////////////////

/// Standard Imports ///
#include <cmath>// ---- `exp` 
using std::isnan;
#include <algorithm> // `clamp`
using std::clamp;
#include <set>
using std::set;

/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;

/// Local ///
#include "utils.hpp"
#include "MNISTBuffer.hpp"


////////// BAYESIAN NEURAL NETWORK /////////////////////////////////////////////////////////////////


struct BayesNeuralLayer{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Bayes Neural Network layer 

    ///// Members / Params ///////////////////////

    uint     dI; // ----- Input  dimensions 
    uint     dIp1; // --- Input  dimensions + 1 (bias)
    uint     dO; // ----- Output dimensions
    MatrixXd x; // ------ Input  values
    MatrixXd lossInp; //- Loss to the previous layer
    MatrixXd y; // ------ Output values
    MatrixXd z; // ------ Output variance
    MatrixXd ySamples; // Output samples
    MatrixXd lossOut; //- Loss from the following layer
    MatrixXd W; // ------ Weight matrix
    double   lr; // ----- Learning rate
    uint     Ninpt; // -- How many samples from the previous layer
    uint     Nsmpl; // -- How many samples to generate per previous sample
    double   initVar; //- Default variance for all activations


    ///// Constructors ///////////////////////////

    BayesNeuralLayer( 
        uint inputDim, uint outputDim, double learnRate, uint upstreamInputs, uint samples
    ){
        // Set params
        dI /**/ = inputDim; // --- Input  dimensions 
        dIp1    = inputDim + 1; // Input  dimensions plus 1 for bias
        dO /**/ = outputDim; // -- Output dimensions
        lr /**/ = learnRate; // -- Learning rate
        Ninpt   = upstreamInputs;
        Nsmpl   = samples;  // --- How many samples to generate
        initVar = 1.0; //- Default variance for all activations

        // Init I/O
        x /*-*/  = MatrixXd{ dIp1, 1 };
        lossInp  = MatrixXd{ dIp1, 1 };
        y /*-*/  = MatrixXd{ dO  , 1 };
        z /*-*/  = MatrixXd{ dO  , 1 };
        lossOut  = MatrixXd{ dO  , 1 };
        ySamples = MatrixXd{ dO  , Ninpt*Nsmpl };

        // Init weights && Gradient
        W = MatrixXd{ dO, dIp1 }; // Weight matrix

        // Bias is unity input at last index
        x( dI, 0 ) = 1.0;
        lossInp( dI, 0 ) = 1.0;
    }

    void random_weight_init( double lo = 0.0f, double hi = 1.0f ){
    // Set all weights and biases to uniformly-distributed random numbers
    // NOTE: Paper does not give a weight init strat
        for( uint i = 0; i < dO; i++ ){
            for( uint j = 0; j < dIp1; j++ ){
                W(i,j) = randd( lo, hi );
            }
            z(i,0) = initVar; // Default variance for all activations
        }
    }

    ///// Inference //////////////////////////

    void load_input( const vf& x_t ){
        // Load values into the input vector
        // NOTE: This struct does not require the client code to add the unity input bias
        for( uint i = 0; i < dI; i++ ){  x(i,0) = x_t[i];  }
    }

    void stochastic_forward( const MatrixXd& inputs ){
        for( uint i = 0; i < Ninpt; i++ ){
            // FIXME, START HERE: LOAD INPUTS ONE AT A TIME, THEN FORWARD, LAST SAMPLE DIST
        }
    }

};