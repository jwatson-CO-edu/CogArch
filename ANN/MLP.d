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


struct PerceptronLayer{
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

    void forward(){
        // Run forward inference
        y = matx_mult_dyn( W, x );
    }

    void margin_update( float[] x_t, float[] y_t, float mu = 0.0f ){
        // Update weights based on a single training example if it within `mu` of the splitting planes
    }

}