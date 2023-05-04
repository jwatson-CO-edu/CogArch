/*
MLP.cpp
James Watson, 2023-04
Simplest demo of a scratch-built multi-layer perceptron (MLP)
g++ MLP.cpp -std=gnu++17 -I /usr/include/eigen3

WARNING: This implementation is absurdly unoptimized!
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

///// Imports ////////////////////////////////////

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
#include <list>
using std::list, std::advance;
#include <string>
using std::string;
#include <fstream>
using std::ifstream, std::ofstream;
#include <sys/stat.h>
#include <filesystem>
#include <ctype.h> // `isspace`
#include <sstream>
using std::istringstream;

/// Linux Imports ///
#include <endian.h>

/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;

///// Type Defines ///////////////////////////////
typedef vector<float> /*---*/ vf;
typedef vector<vector<float>> vvf;
typedef unsigned char /*---*/ ubyte;

////////// HELPER FUNCTIONS ////////////////////////////////////////////////////////////////////////

template<typename T>
ostream& operator<<( ostream& os , const vector<T>& vec ) { 
    // ostream '<<' operator for generic vectors
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
    // Equality operator for generic vectors
    if( vec1.size() != vec2.size() ){
        return false;
    }else{
        for( size_t i = 0; i < vec1.size(); i++ ){  if( vec1[i] != vec2[i] )  return false;  }
        return true;
    }
}

template<typename T>
vector<T> get_slice( const vector<T>& vec, ulong bgn, ulong end ){
    // Get a slice from a generic vector and return it as a vector of the same
    vector<T> rtnVec;
    for( ulong i = bgn; i < end; i++ ){  rtnVec.push_back( vec[i] );  }
    return rtnVec;
}



////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////

double randd(){
    // Return a pseudo-random number between 0.0 and 1.0
    return  1.0 * rand() / RAND_MAX;
}

double randd( double lo, double hi ){
    // Return a pseudo-random number between `lo` and `hi`
    double span = hi - lo;
    return lo + randd() * span;
}

float randf(){
    // Return a pseudo-random number between 0.0 and 1.0
    return  1.0f * rand() / RAND_MAX;
}

float randf( float lo, float hi ){
    // Return a pseudo-random number between `lo` and `hi`
    float span = hi - lo;
    return lo + randf() * span;
}

void seed_rand(){  srand( time( NULL ) );  } // Provide a clock-based unpredictable seed to RNG

double sigmoid( double x ){  return 1.0 / (1.0 + exp(-x));  }
double ddz_sigmoid( double sigZ ){  return sigZ * ( 1.0 - sigZ );  }


///// RANDOM SAMPLING ////////////////////////////

vf sample_uniform_hypercube( size_t dim, float lo, float hi ){
    // Sample from an N-`dim`ensional hypercube with identical bounds for each dimension
    vf    rtnArr;
    float span = hi - lo;
    for( size_t i = 0; i < dim; i++ ){  rtnArr.push_back( lo + span*randf() );  }
    return rtnArr;
}

float test_hyperplane_point( const vf& planeEQ, const vf& pnt ){
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

vvf gen_hyperplane_dataset( size_t M_samples, float lo, float hi, const vf& planeEQ ){
    // Generate a labeled dataset 
    size_t N_dim = planeEQ.size();
    vvf    dataset;
    vf     sample;
    for( size_t i = 0; i < M_samples; i++ ){
        sample = sample_uniform_hypercube( N_dim, lo, hi );
        sample.back() = test_hyperplane_point( planeEQ, sample );
        dataset.push_back( sample );
    }
    return dataset;
}

vf col_vector_to_cpp_vector( const MatrixXd& colVec ){
    // Copy Eigen column vector to flat C++ vector
    size_t /*--*/ Mrows = colVec.rows();
    vf rtnVec;
    for( size_t i = 0; i < Mrows; i++ ){  rtnVec.push_back( colVec(i,0) );  }
    return rtnVec;
}

MatrixXd cpp_vector_to_col_vector( const vf& cppVec ){
    ulong    N_elem = cppVec.size();
    MatrixXd rtnVec = MatrixXd{ N_elem, 1 };
    for( ulong i = 0; i < N_elem; i++ ){  rtnVec(i,0) = cppVec[i];  }
    return rtnVec;
}

////////// MNIST DATA PARSING //////////////////////////////////////////////////////////////////////

void fetch_next_int( ifstream& file, int* numVar ){
    // Fetch 4 bytes from `buffer` and cast as an `int`
    if( !file.eof() && file.is_open() ){
        file.read( reinterpret_cast<char*>( numVar ), sizeof( int ) );
        *numVar = htobe32( *numVar );
    }
}

void fetch_next_ubyte( ifstream& file, ubyte* numVar  ){
    // Fetch 1 byte from `buffer` and cast as an `int`
    if( !file.eof() && file.is_open() ){
        file.read( reinterpret_cast<char*>( numVar ), sizeof( ubyte ) );
    }
}

struct MNISTBuffer{
    // Simplest container for MNIST data

    // Members //
    ifstream imgFile; // ---- Handle to the image file
    ifstream lblFile; // ---- Handle to the label file
    uint     rows; // ------- Height of each image in pixels
    uint     cols; // ------- Width of each image in pixels
    uint     N; // ---------- Number of examples in this dataset
    uint     Nbyte_img; // -- Number of bytes in one image
    uint     Nbyte_imgHdr; // Number of bytes in the image file header
    uint     Nbyte_lbl; // -- Number of bytes in one label
    uint     Nbyte_lblHdr; // Number of bytes in the label file header
     

    vector<int> fetch_image_header(){
        // Fetch the header info from the file and return as a vector
        vector<int> header;
        int elem;
        imgFile.seekg( 0, std::ios::beg );
        for( ubyte i = 0; i < 4; i++ ){ // WARNING: THE LABEL FILE HEADER IS A DIFFERENT LENGTH!
            fetch_next_int( imgFile, &elem );
            header.push_back( elem );
        }
        return header;
    }

    void seek_to_data( ifstream& file, uint numInts ){
        // Set the imgBuffDex to the first index after the data
        int elem;
        file.seekg( 0, std::ios::beg );
        for( uint i = 0; i < numInts; i++ ){ // WARNING: THE LABEL FILE HEADER IS A DIFFERENT LENGTH!
            fetch_next_int( file, &elem );
        }
    }

    void seek_to_data(){
        // Go to the beginning of the data of both files
        seek_to_data( imgFile, 4 );
        seek_to_data( lblFile, 2 );
    }

    bool seek_to_sample( uint i ){
        // Advance the files to the i^{th} sample, Return true if we got there
        if( i >= N ){  
            return false;
        }else{
            imgFile.seekg( Nbyte_imgHdr + i*Nbyte_img );
            lblFile.seekg( Nbyte_lblHdr + i*Nbyte_lbl );
            if((imgFile.peek() == EOF) || (lblFile.peek() == EOF))  
                return false;
            else
                return true;
        }
    }

    vector<uint> generate_random_ordering_of_samples(){
        // Generate and return a vector of randomized indices to access during the epoch
        list<uint> /*-----*/ ordered; //- DLL for O(1) removal
        vector<uint> /*---*/ rndmOrd; //- Vector of indices for the epoch
        list<uint>::iterator it; // ----- Iterator to fetch and pop from
        uint /*-----------*/ fetchDex; // Index of the index to add to the vector
        for( uint i = 0; i < N; i++ ){  ordered.push_back(i);  } // For 0 to N-1, Add the index to the DLL
        for( uint i = 0; i < N; i++ ){ // For 0 to N-1, pop an index from the DLL and add it to the vector to return
            it /*-*/ = ordered.begin(); // ------- Go to the beginning of the list
            fetchDex = rand() % ordered.size(); // Generate a random index within the current list size
            advance( it, fetchDex ); // ---------- Go to that index
            rndmOrd.push_back( *it ); // --------- Fetch elem
            ordered.erase( it ); // -------------- Pop elem
        }
        return rndmOrd;
    }

    MNISTBuffer( string imgPath, string lblPath ){
        // Load metadata and seek to the beginning of both data files
        char byte;
        imgFile = ifstream{ imgPath };
        lblFile = ifstream{ lblPath };
        vector<int> imgHeader = fetch_image_header();
        N    = (uint) imgHeader[1];
        rows = (uint) imgHeader[2];
        cols = (uint) imgHeader[3];
        Nbyte_imgHdr = sizeof( int ) * 4;
        Nbyte_lblHdr = sizeof( int ) * 2;
        Nbyte_img    = sizeof( ubyte ) * rows * cols;
        Nbyte_lbl    = sizeof( ubyte );
        // 3. Set buffer indices to the beginning of labeled data
        seek_to_data();
    }

    vvf fetch_next_image(){
        // Fetch one image worth of data as a float matrix and return it
        vvf   image;
        vf    oneRow;
        ubyte pxlVal;
        for( uint i = 0; i < rows; i++ ){
            oneRow.clear();
            for( uint j = 0; j < cols; j++ ){
                fetch_next_ubyte( imgFile, &pxlVal );
                oneRow.push_back( (float) pxlVal / 255.0f );
            }   
            image.push_back( oneRow );
        }
        return image;
    }

    ubyte fetch_next_label(){
        // Fetch one label, should be called with the same cadence as `fetch_next_image`
        ubyte label;
        fetch_next_ubyte( lblFile, &label );
        return label;
    }

    float count_consecutive_fraction(){
        // Determine if the data has been shuffled for us
        seek_to_data();
        ulong Nsame = 0;
        ubyte Ylast = fetch_next_label();
        ubyte Ycurr;
        for( uint i = 1; i < N; i++ ){
            Ycurr = fetch_next_label();
            if( Ycurr == Ylast )  Nsame++;
            Ylast = Ycurr;
        }
        return 1.0f * Nsame / N;
    }

    vf fetch_next_y(){
        // Fetch one label as a vector, should be called with the same cadence as `fetch_next_image`
        vf    rtnVec;
        ubyte label = fetch_next_label();
        for( ubyte i = 0; i < 10; i++ ){
            if( i == label )  rtnVec.push_back( 1.0f );
            else /*-------*/  rtnVec.push_back( 0.0f );
        }
        return rtnVec;
    }

    void print_mnist_digit( const vvf& image ){
        // Display the given MNIST digit to the terminal with very cheap greyscale
        float pxlVal;
        for( uint i = 0; i < rows; i++ ){
            for( uint j = 0; j < cols; j++ ){
                pxlVal = image[i][j];
                if( pxlVal > 0.75f ) /**/  cout << "#";
                else if( pxlVal > 0.50f )  cout << "*";
                else  /*---------------*/  cout << ".";
                cout << " ";
            }   
            cout << endl;
        }
    }

    void close(){
        imgFile.close();
        lblFile.close();
    }
};



////////// MULTI-LAYERED PERCEPTRON ////////////////////////////////////////////////////////////////

/// Troubleshooting Flag ///
bool _TS_FORWARD = false;
bool _TS_BACKPRP = false;
bool _TS_DATASET = false;

////////// BinaryPerceptronLayer /////////////////

struct BinaryPerceptronLayer{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Perception layer with a binary output vector

    uint     dI; // ----- Input  dimensions 
    uint     dIp1; // --- Input  dimensions + 1 (bias)
    uint     dO; // ----- Output dimensions
    MatrixXd x; // ------ Input  values
    MatrixXd lossInp; //- Loss to the previous layer
    MatrixXd y; // ------ Output values
    MatrixXd lossOut; //- Loss from the following layer
    MatrixXd W; // ------ Weight matrix
    MatrixXd grad; // --- Per-output gradients
    MatrixXd gAcc; // --- Batching gradient accumulator
    double   lr; // ----- Learning rate
    double   rc; // ----- Regularization constant
    double   gs; // ----- Gradient scale
    ulong    Nb; // ----- Batch size

    BinaryPerceptronLayer( 
        uint inputDim, uint outputDim, double learnRate, 
        double lambda = 0.0f, double gradScale = 0.0f
    ){
        // Allocate arrays and init all weights randomly

        // Set params
        dI   = inputDim;
        dIp1 = inputDim + 1; // Input  dimensions plus 1 for bias
        dO   = outputDim; // -- Output dimensions
        lr   = learnRate; // -- Learning rate
        rc   = lambda; // ----- Regularization constant
        gs   = gradScale; // -- Gradient scale

        // Init I/O
        x /*-*/ = MatrixXd{ dIp1, 1 };
        lossInp = MatrixXd{ dIp1, 1 };
        y /*-*/ = MatrixXd{ dO  , 1 };
        lossOut = MatrixXd{ dO  , 1 };

        // Init weights && Gradient
        W    = MatrixXd{ dO, dIp1 };
        grad = MatrixXd{ dO, dIp1 };
        gAcc = MatrixXd::Zero( dO, dIp1 );

        // Bias is unity input at last index
        x( dI, 0 ) = 1.0;
        lossInp( dI, 0 ) = 1.0; // Otherwise this will never be assigned????
    }

    void random_weight_init( double lo = 0.0f, double hi = 1.0f ){
        // Set all weights and biases to uniformly-distributed random numbers
        for( uint i = 0; i < dO; i++ ){
            for( uint j = 0; j < dIp1; j++ ){
                W(i,j) = randd( lo, hi );
            }
        }
    }

    ///// Margin Update //////////////////////////

    MatrixXd forward(){
        // Return a raw double prediction
        return W * x;
    }

    vf forward_vec(){
        // Return a raw double prediction
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

    void load_input( const vf& x_t ){
        // Load values into the input vector
        // NOTE: This struct does not require the client code to add the unity input bias
        for( uint i = 0; i < dI; i++ ){  x(i,0) = x_t[i];  }
    }

    void margin_update( const vf& x_t, const vf& y_t, float mu = 0.0f ){
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

    void train_by_sample( const vvf& samples, float mu = 0.0f ){
        // Train the layer one sample at a time (batch size == 1), No backprop
        size_t M_smpl = samples.size();
        size_t N_cols = samples[0].size();
        vf     x_i;
        vf     y_i;
        for( size_t i = 0; i < M_smpl; i++ ){
            x_i = get_slice<float>( samples[i], 0 , dI     );
            y_i = get_slice<float>( samples[i], dI, N_cols );
            margin_update( x_i, y_i, mu );
        }
    }

    float test_accuracy( const vvf& samples ){
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
        // Return a raw double prediction with sigmoid activation
        y = forward();
        MatrixXd activation = MatrixXd{ dO, 1 };
        for( uint i = 0; i < dO; i++ ){
            y(i,0) /*----*/ = sigmoid( y(i,0) ); 
            activation(i,0) = y(i,0); 
        }
        if( _TS_FORWARD )  cout << "Layer Output:\n" << y << endl;
        return activation;
    }

    vf get_prediction(){
        // Return a vector with the highest weight label as the answer
        vf     yOut;
        double maxVal = -1000.0;
        uint   maxDex = 0;
        for( uint j = 0; j < dO; j++ ){
            if( y(j,0) > maxVal ){
                maxDex = j;
                maxVal = y(j,0);
            }  
        }
        for( uint j = 0; j < dO; j++ ){
            if( j == maxDex )
                yOut.push_back( 1.0f );
            else
                yOut.push_back( 0.0f );
        }
        if( _TS_FORWARD )  cout << "Prediction: " << yOut << endl;
        return yOut;
    }

    void store_output_loss( const vf& y_Actual ){
        // Compute dLoss/dActivation for the OUTPUT layer ONLY
        for( uint i = 0; i < dO; i++ ){
            lossOut(i,0) = 2.0f*( y(i,0) - y_Actual[i] ); // 2*(predicted-desired)
        }
    }

    void add_L1_to_loss(){
        // Add L1 norm of weights to loss
        double accum;
        for( uint i = 0; i < dO; i++ ){
            accum = 0.0f;
            for( uint j = 0; j < dIp1; j++ ){
                accum += abs( W(i,j) );
            }
            lossOut(i,0) += rc * accum;
        }
    }

    void calc_grad(){
        // Calculate the error gradient for the last prediction, given the `y_Actual` labels
        // NOTE: This function assumes that the correct input has already been loaded
        // NOTE: This function assumes that forward inference has already been run
        // double[] y_Predict = predict_sigmoid();
        double dLoss_dAct;
        double dAct_dOut;
        double dOut_dWght;
        // predict_sigmoid( false ); // This already should have been called
        for( uint i = 0; i < dO; i++ ){
            // dLoss_dAct = 2.0f*( y_Predict[i] - y_Actual[i] ); // 2*(predicted-desired)
            // https://youtu.be/Z97XGNUUx9o?t=1520 seems to corroborate the below
            dLoss_dAct = lossOut(i,0); // 2*(predicted-desired)
            dAct_dOut  = y(i,0)*(1.0-y(i,0)); // --- sigmoid(Out)*(1.0-sigmoid(Out))
            for( uint j = 0; j < dIp1; j++ ){
                dOut_dWght = x(j,0);
                grad(i,j) = dOut_dWght * dAct_dOut * dLoss_dAct;
            }
        }
        if( _TS_BACKPRP ){
            cout << "Example Grad Calc: ( " << dOut_dWght << " )( " << dAct_dOut << " )( " << dLoss_dAct << " )" << endl;
        }
    }

    void clear_grad_accum(){
        // Set all elements of the gradient accumulator to zero
        gAcc = MatrixXd::Zero( dO, dIp1 );
    }

    void accum_grad(){
        // Accumulate the current gradient elements
        for( uint i = 0; i < dO; i++ ){
            for( uint j = 0; j < dIp1; j++ ){
                gAcc(i,j) += grad(i,j);
            }
        }
    }

    void set_grad_to_accum(){
        grad = gAcc;
    }

    double grad_norm(){
        // Return the Frobenius norm of the gradient
        return grad.norm();
    }

    void scale_grad(){
        // Scale the gradient to an appropriate magnitude
        double mag = grad.norm();
        grad = (grad / mag) * gs;
    }

    void store_previous_loss(){
        // Calculate loss to be backpropagated to the previous layer
        double accum;
        for( uint j = 0; j < dIp1; j++ ){
            accum = 0.0f;
            for( uint i = 0; i < dO; i++ ){
                // accum += grad(i,j);
                // https://youtu.be/LA4I3cWkp1E?t=2686
                // https://youtu.be/Z97XGNUUx9o?t=1520 corroborates
                accum += grad(i,j) * W(i,j); // 2023-05-02: Slightly better perf w this one
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

    double get_loss(){
        // Get the Manhattan distance between predicted and actual
        double rtnLoss = 0.0f;
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

    double /*--------------------*/ lr; // --------- Learning rate
    double /*--------------------*/ rc; // --------- Regularization constant
    double /*--------------------*/ gs; // --------- Gradient scale
    ulong /*--------------------*/ Nb; // --------- Batch size
    ulong /*--------------------*/ Kb; // --------- Batch counter
    bool /*---------------------*/ useL1reg; // --- Whether to use L1 regularization
    bool /*---------------------*/ pScaleGrad; // - Whether to scale gradients before applying
    bool /*---------------------*/ useMiniBatch; // Whether to use batches
    bool /*---------------------*/ rn; // --------- Whether to randomize epochs
    vector<BinaryPerceptronLayer*> layers; // ----- Dense layers

    MLP( double learnRate, double lambda = 0.0f, double gradScale = 0.0f, ulong batchSize = 0, bool randomizeEpochs = true ){
        // Init hyperparams
        lr = learnRate; // ----- Learning rate
        rc = lambda; // -------- Regularization constant
        gs = gradScale; // ----- Gradient scale
        Nb = batchSize; // ----- Batch size
        Kb = 0; // ------------- Batch counter
        rn = randomizeEpochs; // Whether to randomize epochs
        cout << "Learning Rate: " << lr << endl;
        if( lambda > 0.0f ){  
            useL1reg = true;  
            cout << "L1 Norm will be applied to loss!, lambda = " << rc << endl;
        }else{  useL1reg = false;  }
        if( gradScale > 0.0f ){
            pScaleGrad = true;
            cout << "Each layer's gradient will be scaled to " << gs << endl;
        }else{  pScaleGrad = false;  }
        if( batchSize > 0 ){
            useMiniBatch = true;
            cout << "Using mini-batches of size " << Nb << "!" << endl;
        }
        if( randomizeEpochs ){
            cout << "Data will be shuffled each epoch!" << endl;
        }
    }

    void append_dense_layer( uint inputDim, uint outputDim ){
        // Create a dense layer with specified size, and the learning rate, norm const, and batch size given by parent net
        layers.push_back( new BinaryPerceptronLayer( inputDim, outputDim, lr, rc, gs ) );  
        cout << "Layer " << layers.size() << " created!" << endl;
    }

    void print_arch(){
        // Print the network architecture
        uint i = 1;
        cout << "\n### Network Summary ###" << endl;
        for( BinaryPerceptronLayer* layer : layers ){
            cout << "Layer " << i << ": Input " << layer->dI << "  X  Output " << layer->dO << endl;
            i++;
        }
        cout << "# End of Summary #" << endl << endl;
    }

    vf flatten( const vvf& matx ){
        // Unpack the matrix into a vector by rows
        vf rtnArr;
        for( vf row : matx ){
            for( double elem : row ){
                rtnArr.push_back( elem );
            }
        }
        return rtnArr;
    }

    vf forward( const vvf& matx ){
        // Use the network to run inference on the input image, layer by layer
        for( ulong i = 0; i < layers.size(); i++ ){
            if( i == 0 ){
                layers[i]->load_input( flatten( matx ) );
            }else{
                layers[i]->x.block(0,0,layers[i]->dI,1) = layers[i-1]->y;
            }
            layers[i]->forward_sigmoid();
        }
        return layers.back()->get_prediction();
    }

    void backpropagation( const vf& y_Actual ){
        // Full backpropagation

        long N = layers.size();
        long M;
        layers.back()->store_output_loss( y_Actual );

        if( _TS_BACKPRP ){
            cout << "Output Loss:\n" << layers.back()->lossOut << endl;
        }

        // foreach_reverse( BinaryPerceptronLayer layer; layers[0..$-1] ){
        for( long i = N-1; i > -1; i-- ){
            if( useL1reg ){
                if( _TS_BACKPRP )  cout << endl << "Weight Norm: " << layers[i]->W.norm() << ", Loss before L1: " << layers[i]->lossOut.norm();
                layers[i]->add_L1_to_loss();
                if( _TS_BACKPRP )  cout << ", Loss After L1: " << layers[i]->lossOut.norm() << endl;
            }  
            layers[i]->calc_grad();
            layers[i]->store_previous_loss();
            if( pScaleGrad ){
                layers[i]->scale_grad();
            }
            layers[i]->descend_grad();

            if( i > 0 ){
                M = layers[i-1]->dO;
                layers[i-1]->lossOut.block(0,0,M,1) = layers[i]->lossInp.block(0,0,M,1);
                if( _TS_BACKPRP ){
                    cout << "Layer " << (i+1) << " Input Loss:\n"  << layers[i]->lossInp   << endl;
                    cout << "Layer " << (i  ) << " Output Loss:\n" << layers[i-1]->lossOut << endl;
                }       
            }
        }
    }

    void backup_and_accum( const vf& y_Actual ){
        // Calc gradient, accumulate, and descend batch grad when counter expires
        Kb++;
        long N = layers.size();
        long M;
        layers.back()->store_output_loss( y_Actual );

        if( _TS_BACKPRP ){
            cout << "Output Loss:\n" << layers.back()->lossOut << endl;
        }

        for( long i = N-1; i > -1; i-- ){
            if( useL1reg ){
                if( _TS_BACKPRP )  cout << endl << "Weight Norm: " << layers[i]->W.norm() << ", Loss before L1: " << layers[i]->lossOut.norm();
                layers[i]->add_L1_to_loss();
                if( _TS_BACKPRP )  cout << ", Loss After L1: " << layers[i]->lossOut.norm() << endl;
            }  
            layers[i]->calc_grad();
            layers[i]->accum_grad();
            if( Kb >= Nb ){
                layers[i]->set_grad_to_accum();
                layers[i]->store_previous_loss();
                if( pScaleGrad ){
                    layers[i]->scale_grad();
                }
                layers[i]->descend_grad();
                layers[i]->clear_grad_accum();
            }
            
            if( i > 0 ){
                M = layers[i-1]->dO;
                layers[i-1]->lossOut.block(0,0,M,1) = layers[i]->lossInp.block(0,0,M,1);
                if( _TS_BACKPRP ){
                    cout << "Layer " << (i+1) << " Input Loss:\n"  << layers[i]->lossInp   << endl;
                    cout << "Layer " << (i  ) << " Output Loss:\n" << layers[i-1]->lossOut << endl;
                }       
            }
        }
        if( Kb >= Nb ){
            Kb = 0;
        }
    }

    vf grad_norms(){
        // Return the Frobenius norm of the gradient of each layer, in order
        vf norms;
        for( BinaryPerceptronLayer* layer : layers ){
            norms.push_back( layer->grad_norm() );
        }
        return norms;
    }

    void random_weight_init( double lo = 0.0f, double hi = 1.0f ){
        // Set all weights and biases to uniformly-distributed random numbers
        for( BinaryPerceptronLayer* layer : layers ){
            layer->random_weight_init( lo, hi );
        }
    }

    double get_loss(){
        // Get the Manhattan distance between predicted and actual
        return layers.back()->get_loss();
    }

    double train_one_MNIST_epoch( MNISTBuffer* dataSet ){
        // Run simple backrop (non-batch) on every training example of an `MNISTBuffer`

        // -1. Init vars and reset data buffers
        vvf /*----*/ img; // ---------- Current image
        vf /*-----*/ lbl; // ---------- Current label
        uint /*---*/ N; // ------------ Number of training examples       
        uint /*---*/ div; // ---------- Status print freq 
        double /*--*/ avgLoss = 0.0f; // Average loss for this epoch
        vector<uint> indices;
        if( rn ){  indices = dataSet->generate_random_ordering_of_samples();  }

        N   = dataSet->N;
        div = N/100; // Status print freq 
        
        dataSet->seek_to_data();

        // 0. For every example in the dataset
        for( uint i = 0; i < N; i++ ){

            if( rn ){  dataSet->seek_to_sample( indices[i] );  }

            // 1. Fetch next image and its label
            img.clear();
            lbl.clear();
            img = dataSet->fetch_next_image();
            lbl = dataSet->fetch_next_y();

            // 2. Make prediction and store
            forward( img );          

            // 3. One full backprop step
            if( useMiniBatch )
                backup_and_accum( lbl );
            else
                backpropagation( lbl );

            // 4. Accum loss
            avgLoss += get_loss();

            if( i%div == 0 )  cout << "." << flush;
        }

        cout << "\nGrad Norms: " << grad_norms() << endl;

        // N. Return the average loss across all training example predictions
        return avgLoss / (double) N;
    }
    
    double compare_answers( const vf& pre, const vf& act ){
        // Return true only if `pre` and `act` are identical
        if( pre == act )
            return 1.0f;
        else
            return 0.0f;
    }

    double validate_on_MNIST( MNISTBuffer* dataSet ){
        // Run inference on the validation set and return accuracy

        // -1. Init vars and reset data buffers
        vvf   img; // ------------ Current image
        vf    lbl; // ------------ Current label
        vf    ans; // ------------ Current inference result
        double acc = 0.0f; // ----- Accuracy for this dataset
        uint  N   = dataSet->N; // Number of training examples       
        uint  div = N/100; // ---- Status print freq 
        dataSet->seek_to_data();
        
        cout << "##### VALIDATE #####" << endl;

        // 0. For every example in the dataset
        for( uint i = 0; i < N; i++ ){

            // 1. Fetch next image and its label
            img.clear();
            lbl.clear();
            img = dataSet->fetch_next_image();
            lbl = dataSet->fetch_next_y();

            // 2. Make prediction and store
            ans.clear();
            ans = forward( img );

            // 3. Accum correctness
            acc += compare_answers( ans, lbl );

            if( i%div == 0 )  cout << "." << flush;
        }

        acc /= (double) N;

        cout << endl << "\nValidation Accuracy: " << acc << endl << endl;

        // N. Return accuracy
        return acc;
    }
};


////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

int main(){

    ///// Test 1: Dense Layer ////////////////////
    bool test1 = false;

    if( test1 ){
        // Init perceptron and Train/Test Datasets, 
        BinaryPerceptronLayer bpl /*-*/ = BinaryPerceptronLayer( 3, 1, 0.00001 );
        vf  planeEQ   = {1.0f, 2.0f, 3.0f, 0.0f};
        vvf trainData = gen_hyperplane_dataset( 10000, -5.0f,  5.0f, planeEQ );
        vvf testData  = gen_hyperplane_dataset(  1000, -5.0f,  5.0f, planeEQ );
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

    ///// Test 2: Multiple Layers ////////////////
    bool testRandOrder = false;
    // - 3Blue1Brown Architecture
    //     > Layer 0: Flatten 2D image to 1D vector of 784 elements
    //     > Layer 1: Input 784 --to-> Output  16
    //     > Layer 2: Input  16 --to-> Output  16
    //     > Layer 3: Input  16 --to-> Output  10, Output class for each digit
    MLP net{ 
        0.00002, // 0.000002 // 0.00005 // 0.0001 // 0.00015 // 0.0002 // 0.0003 // 0.0005 // 0.001 // 0.002 // 0.005
        0.00005, // 0.00005 // 0.0002 // 0.0005 // 0.001 // 0.002 // 0.004 // 0.005 // 0.5 // 0.2 // 0.1 // 0.05
        10.0, // 1.0 // 10.0
        10 // 25 // 125 // 250 // 500 // 1000
    }; 
    uint N_epoch = 256; // 128; // 64; // 32 // 16

    if( ! _TS_DATASET ){
        net.append_dense_layer( 784,  32 );
        net.append_dense_layer(  32,  16 );
        net.append_dense_layer(  16,  10 );
        net.random_weight_init( -0.75, +0.75 );  cout << "Weights init!"    << endl;
        net.print_arch();
    }
    
    MNISTBuffer trainDataBuffer{
        "../Data/MNIST/train-images.idx3-ubyte",
        "../Data/MNIST/train-labels.idx1-ubyte"
    }; 
    cout << "Loaded training data!" << endl;
    if( _TS_DATASET ){
        cout << "Training data has " << trainDataBuffer.count_consecutive_fraction() << " consecutive fraction." << endl;
    }


    MNISTBuffer validDataBuffer{ 
        "../Data/MNIST/t10k-images.idx3-ubyte",
        "../Data/MNIST/t10k-labels.idx1-ubyte"
    };  
    cout << "Loaded testing data!" << endl;

    if( testRandOrder ){
        trainDataBuffer.seek_to_sample( 839 );
        trainDataBuffer.print_mnist_digit( trainDataBuffer.fetch_next_image() );
        cout << trainDataBuffer.fetch_next_y() << endl;
        vector<uint> indices = trainDataBuffer.generate_random_ordering_of_samples();
        cout << indices[0] << ", " << indices[5000] << ", " << indices[839] << endl;
    }


    if( _TS_DATASET ){
        cout << "Validation data has " << validDataBuffer.count_consecutive_fraction() << " consecutive fraction." << endl;
    }

    bool testHeader = false;

    if( testHeader ){
        cout << endl << "### Data Fetch Test ###" << endl;

        cout << trainDataBuffer.fetch_image_header() << endl;
        cout << validDataBuffer.fetch_image_header() << endl;

        vvf img = trainDataBuffer.fetch_next_image();
        vf  lbl = trainDataBuffer.fetch_next_y();

        trainDataBuffer.print_mnist_digit( img );
        cout << lbl << endl;

        img = validDataBuffer.fetch_next_image();
        lbl = validDataBuffer.fetch_next_y();

        validDataBuffer.print_mnist_digit( img );
        cout << lbl << endl;

        trainDataBuffer.seek_to_data();
        validDataBuffer.seek_to_data();
    }

    if( _TS_FORWARD || _TS_BACKPRP ){
        // -1. Init vars and reset data buffers
        vvf   img; // --------------- Current image
        vf    lbl; // --------------- Current label
        uint  N; // ----------------- Number of training examples       
        uint  div; // ------- Status print freq 
        double avgLoss = 0.0f; // ---- Average loss for this epoch

        N  = 1;

        // 0. For every example in the dataset
        for( uint i = 0; i < N; i++ ){

            // 1. Fetch next image and its label
            img.clear();
            lbl.clear();
            img = trainDataBuffer.fetch_next_image();
            lbl = trainDataBuffer.fetch_next_y();

            // 2. Make prediction and store
            net.forward( img );          

            // 3. One full backprop step
            net.backpropagation( lbl );
            if( _TS_BACKPRP ){
                cout << "Gradient Norms: " << net.grad_norms() << endl;
            }

            // 4. Accum loss
            avgLoss += net.get_loss();
        }

        cout << "Loss: " << avgLoss << endl;
    }

    ///// Test 2: MNIST //////////////////////////
    bool  test2     = true && ( ! _TS_FORWARD ) && ( ! _TS_DATASET );
    double epochLoss =  0.0f;
    double acc /*-*/ =  0.0f;

    

    if( test2 ){
        cout << "About to train for " << N_epoch << " epochs ..." << endl;
        for( uint i = 0; i < N_epoch; i++ ){
            cout << "##### Epoch " << (i+1) << " #####" << endl;
            epochLoss = net.train_one_MNIST_epoch( &trainDataBuffer );
            cout << endl << "Average loss for one epoch: " << epochLoss << endl << endl;
        }
        cout << "##### Validation #####" << endl;
        acc = net.validate_on_MNIST( &validDataBuffer );
        cout << "Validation Accuracy: " << acc << endl << endl;
    }

    trainDataBuffer.close();
    validDataBuffer.close();

    return 0;
}
