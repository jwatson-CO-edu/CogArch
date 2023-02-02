module RBM;

/*
RBM.d
James Watson, 2023-01
Simplest Restricted Boltzmann Machine (RBM) example
From:
* Hinton, Geoffrey E. "A practical guide to training restricted Boltzmann machines." 
  Neural Networks: Tricks of the Trade: Second Edition (2012): 599-619.
* FIXME: LINK YT LECTURES

rdmd RBM.d
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

import std.stdio; // ---------- `writeln`
import core.stdc.stdlib; // --- `malloc`
import std.math.exponential; // `exp`
import std.random;

// Randomness
Mt19937 rnd;


////////// UTILITY FUNCTIONS ///////////////////////////////////////////////////////////////////////


void[] allocate_zero( size_t size ){
    // allocate_zero a block of untyped bytes that can be managed as a slice.
    // Original Author: Michael Parker, https://dlang.org/blog/2017/09/25/go-your-own-way-part-two-the-heap/

    // malloc(0) is implementation defined (might return null 
    // or an address), but is almost certainly not what we want.
    assert(size != 0);

    void* ptr = calloc( 1, size );
    if(!ptr) assert(0, "Out of memory!");
    
    // Return a slice of the pointer so that the address is coupled
    // with the size of the memory block.
    return ptr[0 .. size];
}


T[][] alloc_2D_array(T)( size_t rows, size_t cols ){
    // allocate_zero a 2D `T` array with `rows*cols` elements and return a pointer to it
    // Original Author: Michael Parker, https://dlang.org/blog/2017/09/25/go-your-own-way-part-two-the-heap/

    // Make sure to account for the size of the
    // array element type!
    return cast(T[][])allocate_zero(T.sizeof * rows * cols); 
}


T[] alloc_array(T)( size_t count ){ 
    // allocate_zero a `T` array with `count` elements and return a pointer to it
    // Original Author: Michael Parker, https://dlang.org/blog/2017/09/25/go-your-own-way-part-two-the-heap/

    // Make sure to account for the size of the
    // array element type!
    return cast(T[])allocate_zero(T.sizeof * count); 
}



////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////

float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }



////////// RESTRICTED BOLTZMANN MACHINE ////////////////////////////////////////////////////////////

struct RBM{
    // Simplest Restricted Boltzmann Machine 

    uint /**/ dI; // Input  dimensions
    uint /**/ dH; // Hidden dimensions
    float[][] W; //- Weight matrix
    float[]   b; //- Hidden  unit biases
    float[]   c; //- Visible unit biases
    float[]   h; //- Hidden  unit values
    float[]   v; //- Visible unit values
    float[]   x; //- Input values
    float     lr; // Learning rate

    this( uint inputDim, uint hiddenDim, float learnRate ){
        // Allocate arrays and init all values to zero

        // Set params
        dI = inputDim;
        dH = hiddenDim;
        lr = learnRate;

        // Init weights
        W = alloc_2D_array!float( dI, dH );
        // Init hidden units
        b = alloc_array!float( dH );
        h = alloc_array!float( dH );
        // Init input units
        c = alloc_array!float( dI );
        v = alloc_array!float( dI );
        x = alloc_array!float( dI );
    }

    void set_input( float[] input ){
        // Populate the input vector
        for( uint k = 0; k < dI; k++ ){
            x[k] = input[k];
        }
    }

    void load_visible(){
        // Populate the visible units with the input vector
        for( uint k = 0; k < dI; k++ ){
            v[k] = x[k];
        }

    }

    float energy(){
        // Calc the (binary) energy function given the current network state
        float netNRG = 0.0;
        float inpNRG = 0.0;
        float hidNRG = 0.0;

        for( uint j = 0; j < dH; j++ ){
            for( uint k = 0; k < dI; k++ ){
                netNRG -= W[k][j] * h[j] * v[k];
            }
            hidNRG -= b[j] * h[j];
        }

        for( uint k = 0; k < dI; k++ ){
            inpNRG -= c[k] * v[k];
        }

        return netNRG + inpNRG + hidNRG;
    }

    float p_h_given_x(){
        // Conditional probabolity of the current hidden values given the current input
        float rtnProd = 1.0;
        float dotProd = 0.0;
        for( uint j = 0; j < dH; j++ ){
            dotProd = 0.0;
            for( uint k = 0; k < dI; k++ ){
                dotProd += W[k][j] * v[k];
            }
            rtnProd *= sigmoid( b[j] + dotProd );
        }
        return rtnProd;
    }

    float p_x_given_h(){
        // Conditional probabolity of the current input given the current hidden values
        float rtnProd = 1.0;
        float dotProd = 0.0;
        for( uint k = 0; k < dI; k++ ){
            dotProd = 0.0;
            for( uint j = 0; j < dH; j++ ){    
                dotProd += W[k][j] * h[j];
            }
            rtnProd *= sigmoid( c[k] + dotProd );
        }
        return rtnProd;
    }

    void Gibbs_sample_visible(){
        // Run Gibbs sampling on the visibile units for `n` steps, `n` wraps
        float p_l = 0.0;

        for( uint k = 0; k < dI; k++ ){
            p_l = c[k];
            for( uint j = 0; j < dH; j++ ){     
                p_l += h[j] * W[k][j];
            }

            if( uniform( 0.0f, 1.0f, rnd ) <= p_l )
                v[k] = 1.0;
            else
                v[k] = 0.0;
        }
    }


    void Gibbs_sample_hidden(){
        // Update the hidden neurons

        /* Section 3.1: Assuming that the hidden units are binary and that you are using Contrastive Divergence, 
        the hidden units should have stochastic binary states when they are being driven by a data-vector. 
        The probability of turning on ahiddenunit,j, is computed by applying the logistic function */

        float p_j = 0.0;

        for( uint j = 0; j < dH; j++ ){    
            p_j = b[j];
            for( uint k = 0; k < dI; k++ ){
                p_j += v[k] * W[k][j];
            }

            if( uniform( 0.0f, 1.0f, rnd ) <= p_j )
                h[j] = 1.0;
            else
                h[j] = 0.0;
        }

    }

    void Contrastive_Divergence_iter(){
        // Run a single iteration of (Persistent) Contrastive Divergence
        
        // 1. Generate a negative sample using n steps of Gibbs sampling
        /* One iteration of alternating Gibbs sampling consists of updating all of the hidden units 
        in parallel followed by updating all of the visible units in parallel. */
        Gibbs_sample_hidden();
        Gibbs_sample_visible();

        // 2. Update parameters
        // FIXME : THIS IS WRONG
        for( uint j = 0; j < dH; j++ ){
            for( uint k = 0; k < dI; k++ ){
                W[k][j] += lr * (h[j]*x[k]*x[k] - h[j]*v[k]*v[k]);
            }
            
        }
    }
}


////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

void main(){
    rnd = Random( unpredictableSeed );

}