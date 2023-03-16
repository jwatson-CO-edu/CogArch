// rdmd af_parseMNIST.d
// http://yann.lecun.com/exdb/mnist/

import std.stdio; // -------------- `writeln`
import std.file; // -------------- `read`
import std.conv;
import std.bitmanip;

struct MNISTBuffer{
    // Simplest container for MNIST data
    ubyte[] buffer;
    size_t  marker;
    uint    rows;
    uint    cols;
    
    this( string fName ){
        // Init struct for returning images
        buffer = cast(ubyte[]) read( fName );
        marker = 0;
        int[] header = fetch_header();
        rows = cast(uint) header[2];
        cols = cast(uint) header[3];
        seek_to_data();
    }

    int fetch_next_int(){
        // Fetch 4 bytes from `buffer` and cast as an `int`
        int rtnVal = peek!(int, Endian.bigEndian)(buffer[marker..$]);
        marker += 4;
        return rtnVal;
    }

    ubyte fetch_next_ubyte(){
        // Fetch 1 byte from `buffer` and cast as a `ubyte`
        ubyte rtnVal = peek!(ubyte, Endian.bigEndian)(buffer[marker..$]);
        marker += 1;
        return rtnVal;
    }

    int[] fetch_header(){
        // Fetch the header info from the file and return as a vector
        int[] header;
        size_t lastMrkr = marker;
        marker = 0;
        for( ubyte i = 0; i < 4; i++ ){
            header ~= fetch_next_int();
        }
        marker = lastMrkr;
        return header;
    }

    void seek_to_data(){
        // Set the marker to the first index after the data
        marker = 4*4;
    }

    float[][] fetch_next_image(){
        // Fetch one image worth of data as a float matrix and return it
        float[][] image;
        float[]   oneRow;
        ubyte     pxlVal;
        for( uint i = 0; i < rows; i++ ){
            oneRow = [];
            for( uint j = 0; j < cols; j++ ){
                pxlVal = fetch_next_ubyte();
                oneRow ~= cast(float) pxlVal / 255.0f;
            }   
            image ~= oneRow;
        }
        return image;
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
            }   
            writeln();
        }
    }
}



void main(){
    MNISTBuffer mb = MNISTBuffer( "../Data/MNIST/train-images.idx3-ubyte" );
    // ubyte[] buffer = cast(ubyte[]) read( "../Data/MNIST/train-images.idx3-ubyte" );
    writeln( mb.buffer.length );
    writeln( mb.fetch_header() );
    writeln( mb.marker );
    mb.seek_to_data();
    writeln( mb.marker );
    writeln( mb.rows );
    writeln( mb.cols );

    
}