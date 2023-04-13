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

float randf( float lo, float hi ){
    // Return a pseudo-random number between `lo` and `hi`
    float span = hi - lo;
    return lo + randf() * span;
}

void seed_rand(){  srand( time( NULL ) );  } // Provide a clock-based unpredictable seed to RNG

float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }
float ddz_sigmoid( float sigZ ){  return sigZ * ( 1.0 - sigZ );  }


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
    // vector<ubyte> imgBuffer; //- Byte buffer of handwritten digit images
    // size_t /*--*/ imgBuffDex; // Points to the next byte to read in the image buffer
    // vector<ubyte> lblBuffer; //- Byte buffer of image labels
    // size_t /*--*/ lblBuffDex; // Points to the next byte to read in the label buffer
    ifstream imgFile;
    ifstream lblFile;
    uint     rows; // ----- Height of each image in pixels
    uint     cols; // ----- Width of each image in pixels
    uint     N; // -------- Number of examples in this dataset

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

    MNISTBuffer( string imgPath, string lblPath ){
        // Load metadata and seek to the beginning of both data files
        char byte;
        imgFile = ifstream{ imgPath };
        lblFile = ifstream{ lblPath };
        vector<int> imgHeader = fetch_image_header();
        N    = (uint) imgHeader[1];
        rows = (uint) imgHeader[2];
        cols = (uint) imgHeader[3];
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
    float    lr; // ----- Learning rate
    float    rc; // ----- Regularization constant

    BinaryPerceptronLayer( uint inputDim, uint outputDim, float learnRate, float lambda = 0.0f ){
        // Allocate arrays and init all weights randomly

        // Set params
        dI   = inputDim;
        dIp1 = inputDim + 1; // Input  dimensions plus 1 for bias
        dO   = outputDim; // -- Output dimensions
        lr   = learnRate; // -- Learning rate
        rc   = lambda; // ----- Regularization constant

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

    void random_weight_init( float lo = 0.0f, float hi = 1.0f ){
        // Set all weights and biases to uniformly-distributed random numbers
        for( uint i = 0; i < dO; i++ ){
            for( uint j = 0; j < dIp1; j++ ){
                W(i,j) = randf( lo, hi );
            }
        }
    }

    ///// Margin Update //////////////////////////

    MatrixXd forward(){
        // Return a raw float prediction
        return W * x;
    }

    vf forward_vec(){
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

    void load_input( const vf& x_t ){
        // Load values into the input vector
        // NOTE: This struct does not require the client code to add the unity input bias
        for( uint i = 0; i < (dIp1-1); i++ ){  x(i,0) = x_t[i];  }
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
        // Return a raw float prediction with sigmoid activation
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
        vf    yOut;
        float maxVal = -1.0f;
        uint  maxDex = 0;
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
        float accum;
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
                dOut_dWght = x(j,0);
                grad(i,j) = dOut_dWght * dAct_dOut * dLoss_dAct;
            }
        }
        if( _TS_BACKPRP ){
            cout << "Example Grad Calc: ( " << dOut_dWght << " )( " << dAct_dOut << " )( " << dLoss_dAct << " )" << endl;
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
    float /*--------------------*/ rc; // --- Regularization constant
    bool /*---------------------*/ useL1reg; // Whether to use L1 regularization
    vector<BinaryPerceptronLayer*> layers; // Dense layers

    MLP( float learnRate, float lambda = 0.0f ){
        // Init hyperparams
        lr = learnRate;
        rc = lambda;
        if( lambda > 0.0f ){  
            useL1reg = true;  
            cout << "L1 Norm will be applied to loss!" << endl;
        }else{  useL1reg = false;  }
    }

    vf flatten( const vvf& matx ){
        // Unpack the matrix into a vector by rows
        vf rtnArr;
        for( vf row : matx ){
            for( float elem : row ){
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
            layers[i]->descend_grad();
            layers[i]->store_previous_loss();

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

    vf grad_norms(){
        // Return the Frobenius norm of the gradient of each layer, in order
        vf norms;
        for( BinaryPerceptronLayer* layer : layers ){
            norms.push_back( layer->grad_norm() );
        }
        return norms;
    }

    void random_weight_init( float lo = 0.0f, float hi = 1.0f ){
        // Set all weights and biases to uniformly-distributed random numbers
        for( BinaryPerceptronLayer* layer : layers ){
            layer->random_weight_init( lo, hi );
        }
    }

    float get_loss(){
        // Get the Manhattan distance between predicted and actual
        return layers.back()->get_loss();
    }

    float train_one_MNIST_epoch( MNISTBuffer* dataSet ){
        // Run simple backrop (non-batch) on every training example of an `MNISTBuffer`

        // -1. Init vars and reset data buffers
        vvf   img; // ---------- Current image
        vf    lbl; // ---------- Current label
        uint  N; // ------------ Number of training examples       
        uint  div; // ---------- Status print freq 
        float avgLoss = 0.0f; // Average loss for this epoch

        N   = dataSet->N;
        div = N/100; // Status print freq 
        
        dataSet->seek_to_data();

        // 0. For every example in the dataset
        for( uint i = 0; i < N; i++ ){

            // 1. Fetch next image and its label
            img.clear();
            lbl.clear();
            img = dataSet->fetch_next_image();
            lbl = dataSet->fetch_next_y();

            // 2. Make prediction and store
            forward( img );          

            // 3. One full backprop step
            backpropagation( lbl );

            // 4. Accum loss
            avgLoss += get_loss();

            if( i%div == 0 )  cout << "." << flush;
        }

        // N. Return the average loss across all training example predictions
        return avgLoss / (float) N;
    }
    
    float compare_answers( const vf& pre, const vf& act ){
        // Return true only if `pre` and `act` are identical
        if( pre == act )
            return 1.0f;
        else
            return 0.0f;
    }

    float validate_on_MNIST( MNISTBuffer* dataSet ){
        // Run inference on the validation set and return accuracy

        // -1. Init vars and reset data buffers
        vvf   img; // ------------ Current image
        vf    lbl; // ------------ Current label
        vf    ans; // ------------ Current inference result
        float acc = 0.0f; // ----- Accuracy for this dataset
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

        acc /= (float) N;

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
    // - 3Blue1Brown Architecture
    //     > Layer 0: Flatten 2D image to 1D vector of 784 elements
    //     > Layer 1: Input 784 --to-> Output  16
    //     > Layer 2: Input  16 --to-> Output  16
    //     > Layer 3: Input  16 --to-> Output  10, Output class for each digit
    MLP net{ 
        0.002, // 0.0001 // 0.00015 // 0.0002 // 0.0003 // 0.0005 // 0.001 // 0.002 // 0.005
        0.005 // 0.00005 // 0.0002 // 0.0005 // 0.001 // 0.002 // 0.004 // 0.005 // 0.5 // 0.2 // 0.1 // 0.05
    }; 
    uint N_epoch   = 32; // 64; // 32 // 16

    net.layers.push_back( new BinaryPerceptronLayer( 784, 16, net.lr, net.rc ) );  cout << "Layer 1 created!" << endl;
    net.layers.push_back( new BinaryPerceptronLayer(  16, 16, net.lr, net.rc ) );  cout << "Layer 2 created!" << endl;
    net.layers.push_back( new BinaryPerceptronLayer(  16, 10, net.lr, net.rc ) );  cout << "Layer 3 created!" << endl;
    net.random_weight_init( 0.001, 0.1 ); /*------------------------------------*/ cout << "Weights init!"    << endl;

    MNISTBuffer trainDataBuffer{
        "../Data/MNIST/train-images.idx3-ubyte",
        "../Data/MNIST/train-labels.idx1-ubyte"
    }; 
    cout << "Loaded training data!" << endl;

    MNISTBuffer validDataBuffer{ 
        "../Data/MNIST/t10k-images.idx3-ubyte",
        "../Data/MNIST/t10k-labels.idx1-ubyte"
    };  
    cout << "Loaded testing data!" << endl;

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
        float avgLoss = 0.0f; // ---- Average loss for this epoch

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
    bool  test2     = true && ( ! _TS_FORWARD );
    float epochLoss =  0.0f;
    float acc /*-*/ =  0.0f;

    if( test2 ){
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
