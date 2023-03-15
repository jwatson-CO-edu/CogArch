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
        // Fetch 4 bytes from a buffer and cast as an `int`
        int rtnVal = peek!(int, Endian.bigEndian)(buffer[marker..$]);
        // int rtnVal = buffer.peek(int,Endian.bigEndian)(&marker);
        marker += 4;
        return rtnVal;
    }

    ubyte fetch_next_byte(){
        // FIXME: GET A SINGLE PIXEL VALUE
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
        // FIXME, START HERE: CONVERT IMAGE TO FLOATS AND RETURN AS A NESTED ARRAY
        // FIXME: PREVENT READING PAST THE END OF THE FILE
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