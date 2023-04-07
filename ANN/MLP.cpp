/*
MLP.cpp
James Watson, 2023-04
Simplest demo of a scratch-built multi-layer perceptron (MLP)
g++ MLP.cpp -std=gnu++17 -I /usr/include/eigen3
*/

/* ///// DEV PLAN /////
[ ] Binary Perceptron @ plane
[ ] Binary MLP w/ Backprop @ MNIST
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

/// Standard Imports ///
#include <cmath>// ---- `exp` 
#include <stdlib.h> //- `srand`, `rand` 
#include <time.h> // -- `time` 
#include <algorithm> // `clamp`
using std::clamp;
#include <iostream>
using std::cout, std::endl, std::flush, std::ostream;
#include <vector>
using std::vector;
#include <set>
using std::set;
#include <string>
using std::string;
#include <fstream>
using std::ifstream, std::ofstream;
#include <sys/stat.h>
#include <filesystem>
#include <ctype.h> // `isspace`
#include <sstream>
using std::istringstream;

/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;



////////// HELPER FUNCTIONS ////////////////////////////////////////////////////////////////////////

template<typename T>
ostream& operator<<( ostream& os , const vector<T>& vec ) { 
    // ostream '<<' operator for vectors
    // NOTE: This function assumes that the ostream '<<' operator for T has already been defined
    os << "[ ";
    for (size_t i = 0; i < vec.size(); i++) {
        os << (T) vec[i];
        if (i + 1 < vec.size()) { os << ", "; }
    }
    os << " ]";
    return os; // You must return a reference to the stream!
}



////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////

float randf(){
    // Return a pseudo-random number between 0.0 and 1.0
    return  1.0f * rand() / RAND_MAX;
}

void seed_rand(){  srand( time( NULL ) );  } // Provide a clock-based unpredictable seed to RNG

float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }
float ddz_sigmoid( float sigZ ){  return sigZ * ( 1.0 - sigZ );  }


///// RANDOM SAMPLING ////////////////////////////

vector<float> sample_uniform_hypercube( size_t dim, float lo, float hi ){
    // Sample from an N-`dim`ensional hypercube with identical bounds for each dimension
    vector<float> rtnArr;
    float /*---*/ span = hi - lo;
    for( size_t i = 0; i < dim; i++ ){  rtnArr.push_back( lo + span*randf() );  }
    return rtnArr;
}

float test_hyperplane_point( const vector<float>& planeEQ, const vector<float>& pnt ){
    // Ground Truth: Return 1.0 if `pnt` is above the plane, otherwise 0.0
    long  N_dim = (long) pnt.size()-1;
    float res   = 0.0;
    for( long i = 0; i < N_dim; i++ ){
        res += pnt[i] * planeEQ[i];
    }
    if( res >= planeEQ.back() )
        return 1.0f;
    else
        return 0.0f;
}

vector<vector<float>> gen_hyperplane_dataset( size_t M_samples, float lo, float hi, const vector<float>& planeEQ ){
    // Generate a labeled dataset 
    size_t /*----------*/ N_dim = planeEQ.size();
    size_t /*----------*/ Nm1   = N_dim-1;
    vector<vector<float>> dataset;
    vector<float> /*---*/ sample;
    for( size_t i = 0; i < M_samples; i++ ){
        sample = sample_uniform_hypercube( Nm1, lo, hi );
        sample.back() = test_hyperplane_point( planeEQ, sample );
        dataset.push_back( sample );
    }
    return dataset;
}

////////// MULTI-LAYERED PERCEPTRON ////////////////////////////////////////////////////////////////


////////// BinaryPerceptronLayer /////////////////


struct BinaryPerceptronLayer{
    // Simplest Perception layer with a binary output vector

    uint     dIp1; // -- Input  dimensions + 1 (bias)
    uint     dO; // ---- Output dimensions
    MatrixXd x; // ----- Input  values
    MatrixXd lossInp; // Loss to the previous layer
    MatrixXd y; // ----- Output values
    MatrixXd lossOut; // Loss from the following layer
    MatrixXd W; // ----- Weight matrix
    MatrixXd grad; // -- Per-output gradients
    float    lr; // ---- Learning rate

    BinaryPerceptronLayer( uint inputDim, uint outputDim, float learnRate ){
        // Allocate arrays and init all weights randomly

        // Set params
        dIp1 = inputDim + 1; // Input  dimensions plus 1 for bias
        dO   = outputDim; // -- Output dimensions
        lr   = learnRate; // -- Learning rate

        // Init I/O
        x /*-*/ = MatrixXd{ dIp1, 1  };
        lossInp = MatrixXd{ dIp1, 1  };
        y /*-*/ = MatrixXd{ 1   , dO };
        lossOut = MatrixXd{ 1   , dO };

        // Init weights && Gradient
        W    = MatrixXd{ dIp1, dO };
        grad = MatrixXd{ dIp1, dO };

        // Bias is unity input at last index
        x( dIp1-1, 0 ) = 1.0f;
    }

    void random_weight_init(){
        // Set all weights and biases to uniformly-distributed random numbers
        for( uint i = 0; i < dIp1; i++ ){
            for( uint j = 0; j < dO; j++ ){
                W(i,j) = randf();
            }
        }
    }

    ///// Margin Update //////////////////////////

    // FIXME, START HERE: ADD MARGIN UPDATE FUNCTIONS

};