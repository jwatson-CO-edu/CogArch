module MLP;

/*
MLP.d
James Watson, 2023-02
Simplest demo of a scratch-built multi-layer perceptron (MLP)
*/

/* ///// DEV PLAN /////
[ ] Binary Perceptron @ plane
[ ] Binary MLP w/ Backprop @ MNIST
[ ] Attempt w/ LAPACK/BLAS
    * https://code.dlang.org/packages/mir
    * https://code.dlang.org/packages/mir-blas
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

import std.random; // RNG

import utils; // - Memory allocation
import mathkit; // (Slow!) Linalg


/// Randomness ///
Mt19937 rnd;


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
    res += planeEQ[$-1];
    if( pnt[$-1] >= res )
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
    float     lr; // - Learning rate

    this( uint inputDim, uint outputDim, float learnRate ){
        // Allocate arrays and init all values to zero

        // Set params
        dIp1 = inputDim + 1;
        dO   = outputDim;
        lr   = learnRate;

        // Init weights
        W = alloc_2D_dyn_array!float( dIp1, dO );

        // Init I/O
        x = alloc_dyn_array!float( dIp1 );
        y = alloc_dyn_array!float( dO   );
        // Bias times unity input
        x[$-1] = 1.0f;
    }


    void random_weight_init(){
        // Set all weights and biases to normally-distributed random numbers
        for( uint i = 0; i < dIp1; i++ ){
            for( uint j = 0; j < dO; j++ ){
                W[i][j] = uniform( 0.0f, 1.0f, rnd );
            }
        }
    }


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
        
        for( uint j = 0; j < dO; j++ ){
            if( (y[j] != y_t[j]) || ( (mu > 0.0f) && (y_p[j] < mu) ) ){
                factor = (y_t[j] == 1.0) ? 1.0 : -1.0f;
                for( uint i = 0; i < (dIp1-1); i++ ){
                    W[i][j] += x_t[i] * lr * factor;
                }
                W[$-1][j] += lr * factor;
            }
        }
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
}

////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

void main(){

    // Init perceptron and Train/Test Datasets, 
    BinaryPerceptronLayer bpl /*-*/ = BinaryPerceptronLayer( 3, 1, 0.0001 );
    float[] /*---------*/ planeEQ   = [1.0f, 2.0f, 3.0f, 0.0f];
    float[][] /*-------*/ trainData = gen_hyperplane_dataset( 1000, -5.0f,  5.0f, planeEQ );
    float[][] /*-------*/ testData  = gen_hyperplane_dataset(  100, -5.0f,  5.0f, planeEQ );
    bpl.random_weight_init();

    // Train //
    // FIXME: TRAIN

    // Test //
    // FIXME: TEST

}
