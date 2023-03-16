// rdmd af_parseMNIST.d
// http://yann.lecun.com/exdb/mnist/

import std.stdio; // -------------- `writeln`
import std.file; // -------------- `read`
import std.conv;
import std.bitmanip;

struct MNISTBuffer{
    // Simplest container for MNIST data
    ubyte[] imgBuffer;
    size_t  imgBuffDex;
    ubyte[] lblBuffer;
    size_t  lblBuffDex;
    uint    rows;
    uint    cols;
    
    this( string imgFName, string lblFName ){
        // Init struct for returning images
        imgBuffer  = cast(ubyte[]) read( imgFName );
        imgBuffDex = 0;
        lblBuffer  = cast(ubyte[]) read( lblFName );
        lblBuffDex = 0;
        int[] header = fetch_header( imgBuffer, &imgBuffDex );
        rows = cast(uint) header[2];
        cols = cast(uint) header[3];
        seek_to_data();
    }

    int fetch_next_int( ubyte[] buffer, size_t* buffDex ){
        // Fetch 4 bytes from `buffer` and cast as an `int`
        int rtnVal = buffer.peek!(int, Endian.bigEndian)(buffDex);
        return rtnVal;
    }

    ubyte fetch_next_ubyte( ubyte[] buffer, size_t* buffDex ){
        // Fetch 1 byte from `buffer` and cast as a `ubyte`
        ubyte rtnVal = buffer.peek!(ubyte, Endian.bigEndian)(buffDex);
        return rtnVal;
    }

    int[] fetch_header( ubyte[] buffer, size_t* buffDex ){
        // Fetch the header info from the file and return as a vector
        int[] header;
        size_t lastMrkr = *buffDex;
        *buffDex = 0;
        for( ubyte i = 0; i < 4; i++ ){
            header ~= fetch_next_int( buffer, buffDex );
        }
        *buffDex = lastMrkr;
        return header;
    }

    void seek_to_data(){
        // Set the imgBuffDex to the first index after the data
        imgBuffDex = 4*4;
        lblBuffDex = 4*2;
    }

    float[][] fetch_next_image(){
        // Fetch one image worth of data as a float matrix and return it
        float[][] image;
        float[]   oneRow;
        ubyte     pxlVal;
        for( uint i = 0; i < rows; i++ ){
            oneRow = [];
            for( uint j = 0; j < cols; j++ ){
                pxlVal = fetch_next_ubyte( imgBuffer, &imgBuffDex );
                oneRow ~= cast(float) pxlVal / 255.0f;
            }   
            image ~= oneRow;
        }
        return image;
    }

    ubyte fetch_next_label(){
        // Fetch one label, should be called with the same cadence as `fetch_next_image`
        return fetch_next_ubyte( lblBuffer, &lblBuffDex );
    }

    void print_mnist_digit( float[][] image ){
        // Display the given MNIST digit to the terminal with very cheap greyscale
        float pxlVal;
        for( uint i = 0; i < rows; i++ ){
            for( uint j = 0; j < cols; j++ ){
                pxlVal = image[i][j];
                if( pxlVal > 0.75f )  write( "#" );
                else if( pxlVal > 0.50f )  write( "*" );
                else  write( "." );
                write( " " );
            }   
            writeln();
        }
    }
}



void main(){
    MNISTBuffer mb = MNISTBuffer( 
        "../Data/MNIST/train-images.idx3-ubyte",
        "../Data/MNIST/train-labels.idx1-ubyte"
    );
    // ubyte[] buffer = cast(ubyte[]) read( "../Data/MNIST/train-images.idx3-ubyte" );
    float[][] img = mb.fetch_next_image();
    mb.print_mnist_digit( img );
    writeln( mb.fetch_next_label() );
}