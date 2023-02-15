module MLP;

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

import std.random; // ------------- RNG

import utils;


/// Randomness ///
Mt19937 rnd;


////////// MATRIX MATH /////////////////////////////////////////////////////////////////////////////

float[] matx_mult_dyn( float[][] matx, float[] vect ){
    // Multiply the `matx` by the `vect`
    // WARNING: SLOW

    // -1. Dims check
    if( matx[0].length != vect.length )  throw new ArrayIndexError( 
        "Dimension mismatch!: " ~ matx[0].length.to!string ~ " -vs- " ~ vect.length.to!string
    );

    // 0. Init
    size_t  M /**/ = matx.length;
    size_t  N /**/ = matx[0].length;
    float   colSum = 0.0f;
    float[] rtnVec = alloc_array!float( M );

    // 1. Mult
    for( size_t i = 0; i < M; i++ ){
        colSum = 0.0f;
        for( size_t j = 0; j < N; j++ ){
            colSum += matx[i][j] * vect[j];
        }
        rtnVec[i] = colSum;
    }

    // N. Return
    return rtnVec;
}


////////// MULTI-LAYERED PERCEPTRON ////////////////////////////////////////////////////////////////

// FIXME: ADD BIAS --> dI+1, the last term of every input is always 1.0f

struct BinaryPerceptronLayer{
    uint /**/ dI; // Input  dimensions
    uint /**/ dO; // Output dimensions
    float[]   x; //- Input  values
    float[]   y; //- Output values
    float[][] W; //- Weight matrix
    float     lr; // Learning rate

    this( uint inputDim, uint outputDim, float learnRate ){
        // Allocate arrays and init all values to zero

        // Set params
        dI = inputDim;
        dO = outputDim;
        lr = learnRate;

        // Init weights
        W = alloc_2D_dyn_array!float( dI, dH );

        // Init I/O
        x = alloc_dyn_array!float( dI );
        y = alloc_dyn_array!float( dO );
    }

    void random_weight_init(){
        // Set all weights and biases to normally-distributed random numbers
        for( uint i = 0; i < dI; i++ ){
            for( uint j = 0; j < dO; j++ ){
                W[i][j] = uniform( 0.0f, 1.0f, rnd );
            }
        }
    }

    float[] forward(){
        return matx_mult_dyn( W, x );
    }

    void predict(){
        // Run forward inference
        y = forward();
        for( uint j = 0; j < dO; j++ ){
            if( y[j] > 0.0f )  
                y[j] = 1.0f;
            else  
                y[j] = 0.0f;
        }
    }

    void load_input( float[] x_t ){
        for( uint i = 0; i < dI; i++ ){
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
                for( uint i = 0; i < dI; i++ ){
                    W[i][j] += x_t[i] * lr * factor;
                }
            }
        }
    }

}