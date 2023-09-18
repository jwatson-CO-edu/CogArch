#ifndef MNISTBUFFER_HPP
#define MNISTBUFFER_HPP

#include <fstream>
using std::ifstream, std::ofstream;
#include <vector>
using std::vector;
#include <list>
using std::list, std::advance;
#include <string>
using std::string;
#include <iostream>
using std::cout, std::endl, std::flush, std::ostream;

///// Type Defines ///////////////////////////////
typedef vector<double> /*---*/ vf;
typedef vector<vector<double>> vvf;
typedef unsigned char /*----*/ ubyte;

////////// MNIST DATA PARSING //////////////////////////////////////////////////////////////////////

void fetch_next_int( ifstream& file, int* numVar ){
    // Fetch 4 bytes from `buffer` and cast as an `int`
    if( !file.eof() && file.is_open() ){
        file.read( reinterpret_cast<char*>( numVar ), sizeof( int ) );
        *numVar = htobe32( *numVar );
    }
}

void fetch_next_ubyte( ifstream& file, ubyte* numVar ){
    // Fetch 1 byte from `buffer` and cast as an `int`
    if( !file.eof() && file.is_open() ){
        file.read( reinterpret_cast<char*>( numVar ), sizeof( ubyte ) );
    }
}

struct MNISTBuffer{
    // Simplest container for MNIST data

    // Members //
    ifstream imgFile; // ---- Handle to the image file
    ifstream lblFile; // ---- Handle to the label file
    uint     rows; // ------- Height of each image in pixels
    uint     cols; // ------- Width of each image in pixels
    uint     N; // ---------- Number of examples in this dataset
    uint     Nbyte_img; // -- Number of bytes in one image
    uint     Nbyte_imgHdr; // Number of bytes in the image file header
    uint     Nbyte_lbl; // -- Number of bytes in one label
    uint     Nbyte_lblHdr; // Number of bytes in the label file header
     

    vector<int> fetch_image_header(){
        // Fetch the header info from the file and return as a vector
        vector<int> header;
        int elem;
        imgFile.seekg( 0, std::ios::beg );
        for( ubyte i = 0; i < 4; i++ ){ // WARNING: THE LABEL FILE HEADER IS A DIFFERENT LENGTH!
            fetch_next_int( imgFile, &elem );
            header.push_back( elem );
        }
        return header;
    }

    void seek_to_data( ifstream& file, uint numInts ){
        // Set the imgBuffDex to the first index after the data
        int elem;
        file.seekg( 0, std::ios::beg );
        for( uint i = 0; i < numInts; i++ ){ // WARNING: THE LABEL FILE HEADER IS A DIFFERENT LENGTH!
            fetch_next_int( file, &elem );
        }
    }

    void seek_to_data(){
        // Go to the beginning of the data of both files
        seek_to_data( imgFile, 4 );
        seek_to_data( lblFile, 2 );
    }

    bool seek_to_sample( uint i ){
        // Advance the files to the i^{th} sample, Return true if we got there
        if( i >= N ){  
            return false;
        }else{
            imgFile.seekg( Nbyte_imgHdr + i*Nbyte_img );
            lblFile.seekg( Nbyte_lblHdr + i*Nbyte_lbl );
            if((imgFile.peek() == EOF) || (lblFile.peek() == EOF))  
                return false;
            else
                return true;
        }
    }

    vector<uint> generate_random_ordering_of_samples(){
        // Generate and return a vector of randomized indices to access during the epoch
        list<uint> /*-----*/ ordered; //- DLL for O(1) removal
        vector<uint> /*---*/ rndmOrd; //- Vector of indices for the epoch
        list<uint>::iterator it; // ----- Iterator to fetch and pop from
        uint /*-----------*/ fetchDex; // Index of the index to add to the vector
        for( uint i = 0; i < N; i++ ){  ordered.push_back(i);  } // For 0 to N-1, Add the index to the DLL
        for( uint i = 0; i < N; i++ ){ // For 0 to N-1, pop an index from the DLL and add it to the vector to return
            it /*-*/ = ordered.begin(); // ------- Go to the beginning of the list
            fetchDex = rand() % ordered.size(); // Generate a random index within the current list size
            advance( it, fetchDex ); // ---------- Go to that index
            rndmOrd.push_back( *it ); // --------- Fetch elem
            ordered.erase( it ); // -------------- Pop elem
        }
        return rndmOrd;
    }

    MNISTBuffer( string imgPath, string lblPath ){
        // Load metadata and seek to the beginning of both data files
        char byte;
        imgFile = ifstream{ imgPath };
        lblFile = ifstream{ lblPath };
        vector<int> imgHeader = fetch_image_header();
        N    = (uint) imgHeader[1];
        rows = (uint) imgHeader[2];
        cols = (uint) imgHeader[3];
        Nbyte_imgHdr = sizeof( int ) * 4;
        Nbyte_lblHdr = sizeof( int ) * 2;
        Nbyte_img    = sizeof( ubyte ) * rows * cols;
        Nbyte_lbl    = sizeof( ubyte );
        // 3. Set buffer indices to the beginning of labeled data
        seek_to_data();
    }

    vvf fetch_next_image(){
        // Fetch one image worth of data as a float matrix and return it
        vvf   image;
        vf    oneRow;
        ubyte pxlVal;
        for( uint i = 0; i < rows; i++ ){
            oneRow.clear();
            for( uint j = 0; j < cols; j++ ){
                fetch_next_ubyte( imgFile, &pxlVal );
                oneRow.push_back( (float) pxlVal / 255.0f );
            }   
            image.push_back( oneRow );
        }
        return image;
    }

    ubyte fetch_next_label(){
        // Fetch one label, should be called with the same cadence as `fetch_next_image`
        ubyte label;
        fetch_next_ubyte( lblFile, &label );
        return label;
    }

    float count_consecutive_fraction(){
        // Determine if the data has been shuffled for us
        seek_to_data();
        ulong Nsame = 0;
        ubyte Ylast = fetch_next_label();
        ubyte Ycurr;
        for( uint i = 1; i < N; i++ ){
            Ycurr = fetch_next_label();
            if( Ycurr == Ylast )  Nsame++;
            Ylast = Ycurr;
        }
        return 1.0f * Nsame / N;
    }

    vf fetch_next_y(){
        // Fetch one label as a vector, should be called with the same cadence as `fetch_next_image`
        vf    rtnVec;
        ubyte label = fetch_next_label();
        for( ubyte i = 0; i < 10; i++ ){
            if( i == label )  rtnVec.push_back( 1.0f );
            else /*-------*/  rtnVec.push_back( 0.0f );
        }
        return rtnVec;
    }

    void print_mnist_digit( const vvf& image ){
        // Display the given MNIST digit to the terminal with very cheap greyscale
        float pxlVal;
        for( uint i = 0; i < rows; i++ ){
            for( uint j = 0; j < cols; j++ ){
                pxlVal = image[i][j];
                if( pxlVal > 0.75f ) /**/  cout << "#";
                else if( pxlVal > 0.50f )  cout << "*";
                else  /*---------------*/  cout << ".";
                cout << " ";
            }   
            cout << endl;
        }
    }

    void close(){
        imgFile.close();
        lblFile.close();
    }
};

#endif // MNISTBUFFER_HPP