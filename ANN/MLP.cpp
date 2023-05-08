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
#include <set>
using std::set;
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

/// Local ///
#include "utils.hpp"
#include "MNISTBuffer.hpp"

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

    ///// Inference //////////////////////////

    void load_input( const vf& x_t ){
        // Load values into the input vector
        // NOTE: This struct does not require the client code to add the unity input bias
        for( uint i = 0; i < dI; i++ ){  x(i,0) = x_t[i];  }
    }

    MatrixXd forward(){
        // Return a raw double prediction
        return W * x;
    }

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
        if( batchSize > 1 ){
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

    ///// Test 2: Multiple Layers ////////////////
    bool testRandOrder = false;
    // - 3Blue1Brown Architecture
    //     > Layer 0: Flatten 2D image to 1D vector of 784 elements
    //     > Layer 1: Input 784 --to-> Output  16
    //     > Layer 2: Input  16 --to-> Output  16
    //     > Layer 3: Input  16 --to-> Output  10, Output class for each digit
    MLP net{ 
        0.000001, // 0.000002 // 0.00005 // 0.0001 // 0.00015 // 0.0002 // 0.0003 // 0.0005 // 0.001 // 0.002 // 0.005
        0.00, // 0.00005 // 0.0002 // 0.0005 // 0.001 // 0.002 // 0.004 // 0.005 // 0.5 // 0.2 // 0.1 // 0.05
        10.0, // 1.0 // 10.0
        32 // 25 // 125 // 250 // 500 // 1000
    }; 
    uint N_epoch = 128; // 256; // 128; // 64; // 32 // 16

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
