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

template<typename T>
bool operator==( const vector<T>& vec1, const vector<T>& vec2 ){
    if( vec1.size() != vec2.size() ){
        return false;
    }else{
        for( size_t i = 0; i < vec1.size(); i++ ){  if( vec1[i] != vec2[i] )  return false;  }
        return true;
    }
}

template<typename T>
vector<T> get_slice( const vector<T>& vec, ulong bgn, ulong end ){
    vector<T> rtnVec;
    for( ulong i = bgn; i < end; i++ ){  rtnVec.push_back( vec[i] );  }
    return rtnVec;
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
    // size_t /*----------*/ Nm1   = N_dim-1;
    vector<vector<float>> dataset;
    vector<float> /*---*/ sample;
    for( size_t i = 0; i < M_samples; i++ ){
        sample = sample_uniform_hypercube( N_dim, lo, hi );
        sample.back() = test_hyperplane_point( planeEQ, sample );
        dataset.push_back( sample );
    }
    return dataset;
}

vector<float> col_vector_to_cpp_vector( const MatrixXd& colVec ){
    // Copy Eigen column vector to flat C++ vector
    size_t /*--*/ Mrows = colVec.rows();
    vector<float> rtnVec;
    for( size_t i = 0; i < Mrows; i++ ){  rtnVec.push_back( colVec(i,0) );  }
    return rtnVec;
}

MatrixXd cpp_vector_to_col_vector( const vector<float>& cppVec ){
    ulong    N_elem = cppVec.size();
    MatrixXd rtnVec = MatrixXd{ N_elem, 1 };
    for( ulong i = 0; i < N_elem; i++ ){  rtnVec(i,0) = cppVec[i];  }
    return rtnVec;
}

////////// MULTI-LAYERED PERCEPTRON ////////////////////////////////////////////////////////////////


////////// BinaryPerceptronLayer /////////////////


struct BinaryPerceptronLayer{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Perception layer with a binary output vector

    uint     dI;
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
        dI   = inputDim;
        dIp1 = inputDim + 1; // Input  dimensions plus 1 for bias
        dO   = outputDim; // -- Output dimensions
        lr   = learnRate; // -- Learning rate

        // Init I/O
        x /*-*/ = MatrixXd{ dIp1, 1 };
        lossInp = MatrixXd{ dIp1, 1 };
        y /*-*/ = MatrixXd{ dO  , 1 };
        lossOut = MatrixXd{ dO  , 1 };

        // Init weights && Gradient
        W    = MatrixXd{ dO, dIp1 };
        grad = MatrixXd{ dO, dIp1 };

        // Bias is unity input at last index
        x( dIp1-1, 0 ) = 1.0f;
    }

    void random_weight_init(){
        // Set all weights and biases to uniformly-distributed random numbers
        for( uint i = 0; i < dO; i++ ){
            for( uint j = 0; j < dIp1; j++ ){
                W(i,j) = randf();
            }
        }
    }

    ///// Margin Update //////////////////////////

    MatrixXd forward(){
        // Return a raw float prediction
        return W * x;
    }

    vector<float> forward_vec(){
        // Return a raw float prediction
        return col_vector_to_cpp_vector( forward() );
    }

    void predict(){
        // Run forward inference and store the binary vector in the output
        y = forward();
        for( uint j = 0; j < dO; j++ ){
            if( y(j,0) > 0.0f )  
                y(j,0) = 1.0f;
            else  
                y(j,0) = 0.0f;
        }
    }

    void load_input( const vector<float>& x_t ){
        // Load values into the input vector
        // NOTE: This struct does not require the client code to add the unity input bias
        for( uint i = 0; i < (dIp1-1); i++ ){  x(i,0) = x_t[i];  }
    }

    void margin_update( const vector<float>& x_t, const vector<float>& y_t, float mu = 0.0f ){
        // Update weights based on a single training example if it within `mu` of the splitting planes
        float    factor = 1.0f;
        MatrixXd y_p    = MatrixXd{ dO, 1 };
        
        load_input( x_t );
        if( mu > 0.0f )  y_p = forward();
        predict();
        
        for( uint j = 0; j < dO; j++ ){
            if( (y(j,0) != y_t[j]) || ( (mu > 0.0f) && (y_p(j,0) < mu) ) ){
                factor = (y_t[j] == 1.0) ?  1.0 : -1.0f;
                for( uint i = 0; i < dI; i++ ){
                    W(j,i) += x_t[i] * lr * factor;
                }
                W(j,dI) += lr * factor;
            }
        }
    }

    void train_by_sample( const vector<vector<float>>& samples, float mu = 0.0f ){
        // Train the layer one sample at a time (batch size == 1), No backprop
        size_t /*--*/ M_smpl = samples.size();
        size_t /*--*/ N_cols = samples[0].size();
        vector<float> x_i;
        vector<float> y_i;
        for( size_t i = 0; i < M_smpl; i++ ){
            // cout << "Get slices ..." << endl;
            x_i = get_slice<float>( samples[i], 0 , dI     );
            y_i = get_slice<float>( samples[i], dI, N_cols );
            // cout << "Margin update ..." << endl;
            // cout << x_i << "  ,  " << y_i << "  ,  " << mu << endl;
            margin_update( x_i, y_i, mu );
        }
    }

    float test_accuracy( const vector<vector<float>>& samples ){
        // Test accuracy for the trained layer
        size_t N_smpl = samples.size();
        size_t N_cols = samples[0].size();
        size_t N_crct = 0;
        for( size_t i = 0; i < N_smpl; i++ ){
            // cout << "Load input ..." << endl;
            load_input(  get_slice<float>( samples[i], 0, dI )  );
            // cout << "Predict ..." << endl;
            predict();
            // cout << "Check ..." << endl;
            if( get_slice<float>( samples[i], dI, N_cols ) == col_vector_to_cpp_vector(y) )  N_crct++;
        }
        return ((float) N_crct) / ((float) N_smpl);
    }

    ///// Backpropagation ////////////////////////
    
    // * Overview: https://www.youtube.com/watch?v=Ilg3gGewQ5U
    // * Math: ___ https://www.youtube.com/watch?v=tIeHLnjs5U8
    
    //     - The nudge that we give a weight should be proportional to how far the result was from the target
    //         > Bump up   weights feeding an activation that should be higher
    //         > Bump down weights feeding an activation that should be lower

    //     - Cost = Squared Error = (desired - predicted)^2
    /*

    /// Chain Rule Per Layer Per Training Example ///
    The chain rule describes how sensitive the activation is to the loss
    https://www.youtube.com/watch?v=tIeHLnjs5U8
    
    
    dCost      dOutput     dActivation                         dCost
    -------- = -------  *  -----------                      *  -----------
    dWeights   dWeights    dOutput                             dActivation

    dCost                              
    -------- = (Input)  *  sigmoid(Out)*(1.0-sigmoid(Out))  *  2*(predicted-desired)
    dWeights   

    dCost                              
    -------- = (1.0)    *  sigmoid(Out)*(1.0-sigmoid(Out))  *  2*(predicted-desired)
    dBias 

    */
    
    //     - 3Blue1Brown Architecture
    //         > Layer 0: Flatten 2D image to 1D vector of 784 elements
    //         > Layer 1: Input 784 --to-> Output  16
    //         > Layer 2: Input  16 --to-> Output  16
    //         > Layer 3: Input  16 --to-> Output  10, Output class for each digit

    MatrixXd forward_sigmoid(){
        // Return a raw float prediction with sigmoid activation
        y = forward();
        MatrixXd activation = MatrixXd{ dO, 1 };
        for( uint i = 0; i < dO; i++ ){
            y(i,0) /*----*/ = sigmoid( y(i,0) ); 
            activation(i,0) = y(i,0); 
        }
        return activation;
    }

    vector<float> get_prediction(){
        // Return a vector with the highest weight label as the answer
        vector<float> yOut;
        float /*---*/ maxVal = -1.0f;
        uint /*----*/ maxDex = 0;
        for( uint j = 0; j < dO; j++ ){
            if( y(j,0) > maxVal ){
                maxDex = j;
                maxVal = (j,0);
            }  
        }
        for( uint j = 0; j < dO; j++ ){
            if( j == maxDex )
                yOut.push_back( 1.0f );
            else
                yOut.push_back( 0.0f );
        }
        return yOut;
    }

    void store_output_loss( const vector<float>& y_Actual ){
        // Compute dLoss/dActivation for the OUTPUT layer ONLY
        for( uint i = 0; i < dO; i++ ){
            lossOut(i,0) = 2.0f*( y(i,0) - y_Actual[i] ); // 2*(predicted-desired)
        }
    }

    void calc_grad(){
        // Calculate the error gradient for the last prediction, given the `y_Actual` labels
        // NOTE: This function assumes that the correct input has already been loaded
        // NOTE: This function assumes that forward inference has already been run
        // float[] y_Predict = predict_sigmoid();
        float dLoss_dAct;
        float dAct_dOut;
        float dOut_dWght;
        // predict_sigmoid( false ); // This already should have been called
        for( uint i = 0; i < dO; i++ ){
            // dLoss_dAct = 2.0f*( y_Predict[i] - y_Actual[i] ); // 2*(predicted-desired)
            dLoss_dAct = lossOut(i,0); // 2*(predicted-desired)
            dAct_dOut  = y(i,0)*(1.0-y(i,0)); // --- sigmoid(Out)*(1.0-sigmoid(Out))
            for( uint j = 0; j < dIp1; j++ ){
                dOut_dWght = x[j];
                grad(i,j) = dOut_dWght * dAct_dOut * dLoss_dAct;
            }
        }
    }

    float grad_norm(){
        // Return the Frobenius norm of the gradient
        return grad.norm();
    }

    void store_previous_loss(){
        // Calculate loss to be backpropagated to the previous layer
        float accum;
        for( uint j = 0; j < dIp1; j++ ){
            accum = 0.0f;
            for( uint i = 0; i < dO; i++ ){
                accum += grad(i,j);
            }
            lossInp(j,0) = accum;
        }
    }

    void descend_grad(){
        // Apply one step of gradient descent according to the learning rate
        for( uint i = 0; i < dO; i++ ){
            for( uint j = 0; j < dIp1; j++ ){
                W(i,j) -= lr * grad(i,j);
            }
        }
    }

    float get_loss(){
        // Get the Manhattan distance between predicted and actual
        float rtnLoss = 0.0f;
        for( uint i = 0; i < dO; i++ ){
            rtnLoss += abs( lossOut(i,0) ); // 2*(predicted-desired)
        }
        return rtnLoss / 2.0f;
    }

};

////////// MLP ///////////////////////////////////


struct MLP{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Multi-Layer Perceptron Implementation, Very inefficient
    // WARNING: NO DIMS CHECKED!

    float /*--------------------*/ lr; // --- Learning rate
    vector<BinaryPerceptronLayer*> layers; // Dense layers

    MLP( float learnRate ){
        // Init learn rate
        lr = learnRate;
    }

    vector<float> flatten( const vector<vector<float>>& matx ){
        // Unpack the matrix into a vector by rows
        vector<float> rtnArr;
        for( vector<float> row : matx ){
            for( float elem : row ){
                rtnArr.push_back( elem );
            }
        }
        return rtnArr;
    }

    vector<float> forward( const vector<vector<float>>& matx ){
        // Use the network to run inference on the input image, layer by layer
        for( ulong i = 0; i < layers.size(); i++ ){
            if( i == 0 ){
                layers[i]->load_input( flatten( matx ) );
            }else{
                layers[i]->x = layers[i-1]->y;
            }
            layers[i]->forward_sigmoid();
        }
        return layers.back()->get_prediction();
    }

    void backpropagation( vector<float> y_Actual ){
        // Full backpropagation

        long N = layers.size();
        long M;
        layers.back()->store_output_loss( y_Actual );
        // foreach_reverse( BinaryPerceptronLayer layer; layers[0..$-1] ){
        for( long i = N-1; i > -1; i-- ){
            layers[i]->calc_grad();
            layers[i]->descend_grad();
            layers[i]->store_previous_loss();

            if( i > 0 ){
                M = layers[i-1]->dO;

                // FIXME, START HERE: COMPLETE TRANSLATION

                layers[i-1].lossOut[0..M] = layers[i].lossInp[0..M]; // We can copy arrays by slice

                if( _DEBUG ){
                    writeln( "Copied loss: " ~ layers[i-1].lossOut.to!string );
                    writeln();
                }
            }
        }
    }

};

////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

int main(){

    ///// Test 1: Dense Layer ////////////////////
    bool test1 = true;

    if( test1 ){
        // Init perceptron and Train/Test Datasets, 
        BinaryPerceptronLayer bpl /*-*/ = BinaryPerceptronLayer( 3, 1, 0.00001 );
        vector<float> /*---*/ planeEQ   = {1.0f, 2.0f, 3.0f, 0.0f};
        vector<vector<float>> trainData = gen_hyperplane_dataset( 10000, -5.0f,  5.0f, planeEQ );
        vector<vector<float>> testData  = gen_hyperplane_dataset(  1000, -5.0f,  5.0f, planeEQ );
        bpl.random_weight_init();

        uint N_epoch = 16;

        for( uint i = 0; i < N_epoch; i++ ){

            cout << "Epoch " << (i+1) << ", Test Acc: "; // 0.99

            // Train //
            bpl.train_by_sample( trainData, 3.0f );

            // Test //
            cout << bpl.test_accuracy( testData ) << endl;
            cout << bpl.W << endl;
        }
    }

    return 0;
}
