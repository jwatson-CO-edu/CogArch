module RBM;

/*
RBM.d
James Watson, 2023-01
Simplest Restricted Boltzmann Machine (RBM) example

rdmd RBM.d
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

import std.stdio; // ------ `writeln`
import core.stdc.stdlib; // `malloc`

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


////////// RESTRICTED BOLTZMANN MACHINE ////////////////////////////////////////////////////////////

struct RBM{
    float[][] W; // Weight matrix
    float[]   b; // Hidden unit biases
    float[]   c; // Input  unit biases
    float[]   h; // Hidden unit values
    float[]   x; // Input  unit values

    this( uint inputDim, uint hiddenDim ){
        // Init weights
        W = alloc_2D_array!float( inputDim, hiddenDim );
        b = alloc_array!float( hiddenDim );
        h = alloc_array!float( hiddenDim );
        c = alloc_array!float( inputDim );
        
    }
}


////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

void main(){

}