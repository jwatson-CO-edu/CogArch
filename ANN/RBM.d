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
    float[]   b; //- Hidden unit biases
    float[]   c; //- Input  unit biases
    float[]   h; //- Hidden unit values
    float[]   x; //- Input  unit values
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
        x = alloc_array!float( dI );
    }

    float energy(){
        // Calc the (binary) energy function given the current network state
        float netNRG = 0.0;
        float inpNRG = 0.0;
        float hidNRG = 0.0;

        for( uint j = 0; j < dH; j++ ){
            for( uint k = 0; k < dI; k++ ){
                netNRG -= W[k][j] * h[j] * x[k];
            }
            hidNRG -= b[j] * h[j];
        }

        for( uint k = 0; k < dI; k++ ){
            inpNRG -= c[k] * x[k];
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
                dotProd += W[k][j] * x[k];
            }
            rtnProd *= sigmoid( b[j] + dotProd );
        }
        return rtnProd;
    }

    float p_x_given_h(){
        // Conditional probabolity of the current input given the current hidden values

    }
}


////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

void main(){

}