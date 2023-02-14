module MLP;

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

/// Randomness ///
Mt19937 rnd;



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

}