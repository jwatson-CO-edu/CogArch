module ANN.utils;

import std.stdio;

////////// MEMORY //////////////////////////////////////////////////////////////////////////////////

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



////////// FILE OPS ////////////////////////////////////////////////////////////////////////////////


string[] read_lines( string fName ){
    // Read a file line by line, storing each line in a string vector, Return vector
    File     file = File( fName, "r");
    string   line;
    string[] lines;
    while ( (line = file.readln() ) !is null){
        lines ~= line;
    }
    return lines;
}

T[][] file_to_matx_ws(T)( string fName ){
    // Convert a file to a matrix of `T`, with each line of a the file being a row of the matrix, 
    // cols separated by whitespace
    // NOTE: This function assumes a uniform number of columns across all non-empty lines of the file

    // 1. Read the file
    string[] fLines = read_lines( fName ); // Lines of raw text

    // 2. Determine columns
    string[] oneCol = fLines[0].split!isWhite;
    ulong    nCols  = oneCol.length;

    // 3. Determine rows
    foreach( string fRow; fLines ){ 
        oneCol = fRow.split!isWhite;
        // FIXME, START HERE: COUNT ROWS, REJECT ALL EMPTY ROWS
    }


}

// void main(){
    

// }