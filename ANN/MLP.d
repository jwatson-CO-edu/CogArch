module MLP;

/*
MLP.d
James Watson, 2023-02
Simplest demo of a scratch-built multi-layer perceptron (MLP)
rdmd MLP.d -I ANN/
*/

/* ///// DEV PLAN /////
[Y] Binary Perceptron @ plane - 2023-02-17: Test Acc, 0.977
[ ] Binary MLP w/ Backprop @ MNIST
[ ] Attempt w/ LAPACK/BLAS
    * https://code.dlang.org/packages/mir
    * https://code.dlang.org/packages/mir-blas
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

import std.stdio; //- `writeln`
import std.random; // RNG
import std.conv; //-- `to!string`

import utils; // - Memory allocation
import mathkit; // (Slow!) Linalg


/// Randomness ///
Mt19937 rnd;

////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////


float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }
float ddz_sigmoid( float sigZ ){  return sigZ * ( 1.0 - sigZ );  }

////////// RANDOM SAMPLING /////////////////////////////////////////////////////////////////////////


float[] sample_uniform_hypercube( size_t dim, float lo, float hi ){
    // Sample from an N-`dim`ensional hypercube with identical bounds for each dimension
    float[] rtnArr;
    for( size_t i = 0; i < dim; i++ ){
        rtnArr ~= uniform( lo, hi, rnd );
    }
    return rtnArr;
}


float test_hyperplane_point( float[] planeEQ, float[] pnt ){
    // Ground Truth: Return 1.0 if `pnt` is above the plane, otherwise 0.0
    long   N_dim = cast(long) pnt.length-1;
    float  res   = 0.0;
    for( long i = 0; i < N_dim; i++ ){
        res += pnt[i] * planeEQ[i];
    }
    // res += planeEQ[$-1];
    if( res >= planeEQ[$-1] )
        return 1.0f;
    else
        return 0.0f;
}


float[][] gen_hyperplane_dataset( size_t M_samples, float lo, float hi, float[] planeEQ ){
    // Generate a labeled dataset 
    size_t    N_dim   = planeEQ.length;
    size_t    Nm1     = N_dim-1;
    float[][] dataset = alloc_2D_dyn_array!float( M_samples, N_dim );
    float[]   sample;
    for( size_t i = 0; i < M_samples; i++ ){
        sample = sample_uniform_hypercube( Nm1, lo, hi );
        for( uint j = 0; j < Nm1; j++ ){
            dataset[i][j] = sample[j]; // FIXME: CAN I ASSIGN SLICES?
        }
        dataset[i][$-1] = test_hyperplane_point( planeEQ, sample );
    }
    return dataset;
}


////////// MULTI-LAYERED PERCEPTRON ////////////////////////////////////////////////////////////////


struct BinaryPerceptronLayer{
    // Simplest Perception layer with a binary output vector
    uint /**/ dIp1; // Input  dimensions + 1 (bias)
    uint /**/ dO; //-- Output dimensions
    float[]   x; // -- Input  values
    float[]   y; // -- Output values
    float[][] W; // -- Weight matrix
    float[][] grad; // Per-output gradients
    float     lr; // - Learning rate

    this( uint inputDim, uint outputDim, float learnRate ){
        // Allocate arrays and init all values to zero

        // Set params
        dIp1 = inputDim + 1; // Input  dimensions plus 1 for bias
        dO   = outputDim; // -- Output dimensions
        lr   = learnRate; // -- Learning rate

        // Init weights
        W    = alloc_2D_dyn_array!float( dO, dIp1 ); // Layer weights
        grad = alloc_2D_dyn_array!float( dO, dIp1 ); // Weight gradients

        // Init I/O
        x /*-*/ = alloc_dyn_array!float( dIp1 ); // Input vector
        lossInp = alloc_dyn_array!float( dIp1 ); // Loss to the previous layer
        y /*-*/ = alloc_dyn_array!float( dO   ); // Output vector
        lossOut = alloc_dyn_array!float( dO   ); // Loss from the following layer
        // Bias times unity input
        x[$-1] = 1.0f;
    }


    void random_weight_init(){
        // Set all weights and biases to normally-distributed random numbers
        for( uint i = 0; i < dO; i++ ){
            for( uint j = 0; j < dIp1; j++ ){
                W[i][j] = uniform( 0.0f, 1.0f, rnd );
            }
        }
    }


    ///// Margin Update //////////////////////////

    float[] forward(){
        // Return a raw float prediction
        return matx_mult_dyn( W, x );
    }

    
    void predict(){
        // Run forward inference and store the binary vector in the output
        y = forward();
        for( uint j = 0; j < dO; j++ ){
            if( y[j] > 0.0f )  
                y[j] = 1.0f;
            else  
                y[j] = 0.0f;
        }
    }

    
    void load_input( float[] x_t ){
        // Load values into the input vector
        // NOTE: This struct does not require the client code to add the unity input bias
        for( uint i = 0; i < (dIp1-1); i++ ){
            x[i] = x_t[i];
        }
    }


    void margin_update( float[] x_t, float[] y_t, float mu = 0.0f ){
        // Update weights based on a single training example if it within `mu` of the splitting planes
        float   factor = 1.0f;
        float[] y_p;
        
        load_input( x_t );
        if( mu > 0.0f )  y_p = forward();
        predict();
        
        for( uint j = 0; j < dO; j++ ){
            if( (y[j] != y_t[j]) || ( (mu > 0.0f) && (y_p[j] < mu) ) ){
                factor = (y_t[j] == 1.0) ?  1.0 : -1.0f;
                for( uint i = 0; i < (dIp1-1); i++ ){
                    // write( (x_t[i] * lr * factor).to!string ~ ", " );
                    W[j][i] += x_t[i] * lr * factor;
                }
                W[j][$-1] += lr * factor;
            }
        }
        // writeln();
    }


    void train_by_sample( float[][] samples, float mu = 0.0f ){
        // Train the layer one sample at a time (batch size == 1), No backprop
        size_t  M_smpl = samples.length;
        float[] x_i;
        float[] y_i;
        for( size_t i = 0; i < M_smpl; i++ ){
            x_i = samples[i][0..$-dO];
            y_i = samples[i][$-dO..$];
            margin_update( x_i, y_i, mu );
        }
    }

    float test_accuracy( float[][] samples ){
        size_t N_smpl = samples.length;
        size_t N_crct = 0;
        for( size_t i = 0; i < N_smpl; i++ ){
            load_input( samples[i][0..(dIp1-1)] );
            predict();
            if(samples[i][$-dO..$] == y)  N_crct++;
        }
        return (cast(float) N_crct) / (cast(float) N_smpl);
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

    

    float[] output_loss( float[] yTarget ){
        // Compute derivative of squared loss
        // NOTE: This function assumes that the client code has run `predict()`
        float[] loss;
        for( uint j = 0; j < dO; j++ ){
            loss ~= y[j] - yTarget[j];
        }
        return loss;
    }


    float[] forward_sigmoid(){
        // Return a raw float prediction with sigmoid activation
        float[] product = matx_mult_dyn( W, x );
        float[] activation;
        foreach( float prod; product ){
            activation ~= sigmoid( prod );
        }
        return activation;
    }


    void predict_sigmoid( bool roundBinary = true ){
        // Run forward inference and store the binary vector in the output
        y = forward_sigmoid();
        if( roundBinary ){
            for( uint j = 0; j < dO; j++ ){
                if( y[j] >= 0.5f )  
                    y[j] = 1.0f;
                else  
                    y[j] = 0.0f;
            }
        }
    }


    void store_output_loss( float[] y_Actual ){
        // float[] y_Predict = predict_sigmoid();
        for( uint i = 0; i < dO; i++ ){
            lossOut[i] = 2.0f*( y[i] - y_Actual[i] ); // 2*(predicted-desired)
        }
    }

    void calc_grad(){
        // Calculate the error gradient for the last prediction, given the `y_Actual` labels
        // NOTE: This function assumes that the correct input has already been loaded
        // NOTE: This function assumes that forward inference has already been run
        // float[] y_Predict = predict_sigmoid();
        float   dLoss_dAct;
        float   dAct_dOut;
        float   dOut_dWght;
        // predict_sigmoid( false ); // This already should have been called
        for( uint i = 0; i < dO; i++ ){
            // dLoss_dAct = 2.0f*( y_Predict[i] - y_Actual[i] ); // 2*(predicted-desired)
            dLoss_dAct = lossOut[i]; // 2*(predicted-desired)
            dAct_dOut  = y[i]*(1.0-y[i]); // --- sigmoid(Out)*(1.0-sigmoid(Out))
            for( uint j = 0; j < dIp1; j++ ){
                dOut_dWght = x[j];
                grad[j][i] = dOut_dWght * dAct_dOut * dLoss_dAct;
            }
        }
    }

    void store_previous_loss(){
        // Calculate loss to be backpropagated to the previous layer
        float accum;
        for( uint j = 0; j < dIp1; j++ ){
            accum = 0.0f;
            for( uint i = 0; i < dO; i++ ){
                accum += grad[j][i];
            }
            lossInp[j] = accum;
        }
    }

    void descend_grad(){
        // Apply one step of gradient descent according to the learning rate
        for( uint i = 0; i < dO; i++ ){
            for( uint j = 0; j < dIp1; j++ ){
                W[j][i] -= lr * grad[j][i];
            }
        }
    }
}

struct MLP{
    // Simplest Multi-Layer Perceptron Implementation, Very inefficient
    // WARNING: NO DIMS CHECKED!

    float /*-------------*/ lr; // --- Learning rate
    BinaryPerceptronLayer[] layers; // Dense layers
    // float[] /*-----------*/ X_temp; // Input  to   the current layer
    float[] /*-----------*/ Y_temp; // Output from the current layer

    this( float learnRate ){
        // Init learn rate and memory
        lr     = learnRate;
        layers = [];
    }

    float[] flatten( float[][] matx ){
        // Unpack the matrix into a vector by rows
        float[] rtnArr;
        foreach( float[] row; matx ){
            foreach( float elem; row ){
                rtnArr ~= elem;
            }
        }
        return rtnArr;
    }

    float[] forward( float[][] matx ){
        // Use the network to run inference on the input image, layer by layer
        Y_temp = flatten( matx );
        foreach( BinaryPerceptronLayer layer; layers[0..$-1] ){
            layer.load_input( Y_temp );
            Y_temp = forward_sigmoid();
        }
        layers[$-1].load_input( Y_temp );
        layers[$-1].predict_sigmoid();
        return layers[$-1].y;
    }

    // FIXME, START HERE: FULL BACKPROP
}

////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

void main(){

    ///// Test 1: Dense Layer ////////////////////

    // Init perceptron and Train/Test Datasets, 
    BinaryPerceptronLayer bpl /*-*/ = BinaryPerceptronLayer( 3, 1, 0.00001 );
    float[] /*---------*/ planeEQ   = [1.0f, 2.0f, 3.0f, 0.0f];
    // float[] /*---------*/ planeEQ   = [1.0f, 2.0f, 0.0f];
    float[][] /*-------*/ trainData = gen_hyperplane_dataset( 10000, -5.0f,  5.0f, planeEQ );
    float[][] /*-------*/ testData  = gen_hyperplane_dataset(  1000, -5.0f,  5.0f, planeEQ );
    bpl.random_weight_init();

    uint N_epoch = 64;

    for( uint i = 0; i < N_epoch; i++ ){

        write( "Epoch " ~ (i+1).to!string ~ ", Test Acc: " );

        // Train //
        bpl.train_by_sample( trainData, 3.0f );

        // Test //
        writeln( bpl.test_accuracy( testData ) );
        writeln( bpl.W );
    }
    
    ///// Test 2: Multiple Layers ////////////////

}
