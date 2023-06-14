#ifndef UTILS_HPP
#define UTILS_HPP

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

///// Imports ////////////////////////////////////

/// Standard Imports ///
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl, std::flush, std::ostream;
#include <cmath>// ---- `exp` 
#include <string>
using std::string, std::stod;
#include <sys/stat.h>
#include <fstream>
using std::ifstream, std::ofstream;

///// Type Defines ///////////////////////////////
typedef vector<double> /*---*/ vd;
typedef vector<vector<double>> vvd;
typedef vector<string> /*---*/ vstr;
typedef unsigned char /*----*/ ubyte;

////////// HELPER FUNCTIONS ////////////////////////////////////////////////////////////////////////

template<typename T>
ostream& operator<<( ostream& os , const vector<T>& vec ) { 
    // ostream '<<' operator for generic vectors
    // NOTE: This function assumes that the ostream '<<' operator for T has already been defined
    os << "[ ";
    for (size_t i = 0; i < vec.size(); i++) {
        os << (T) vec[i];
        if (i + 1 < vec.size()) { os << ", "; }
    }
    os << " ]";
    return os; // You must return a reference to the stream!
}

template<typename T>
bool operator==( const vector<T>& vec1, const vector<T>& vec2 ){
    // Equality operator for generic vectors
    if( vec1.size() != vec2.size() ){
        return false;
    }else{
        for( size_t i = 0; i < vec1.size(); i++ ){  if( vec1[i] != vec2[i] )  return false;  }
        return true;
    }
}

////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////

double randd(){
    // Return a pseudo-random number between 0.0 and 1.0
    return  1.0 * rand() / RAND_MAX;
}

double randd( double lo, double hi ){
    // Return a pseudo-random number between `lo` and `hi`
    double span = hi - lo;
    return lo + randd() * span;
}

float randf(){
    // Return a pseudo-random number between 0.0 and 1.0
    return  1.0f * rand() / RAND_MAX;
}

float randf( float lo, float hi ){
    // Return a pseudo-random number between `lo` and `hi`
    float span = hi - lo;
    return lo + randf() * span;
}

void seed_rand(){  srand( time( NULL ) );  } // Provide a clock-based unpredictable seed to RNG

double sigmoid( double x ){  return 1.0 / (1.0 + exp(-x));  }
double ddz_sigmoid( double sigZ ){  return sigZ * ( 1.0 - sigZ );  }

double Box_Muller_normal_sample( double mu = 0.0f, double sigma = 1.0f ){
    // Transform 2 uniform samples into a zero-mean normal sample
    // Source: https://www.baeldung.com/cs/uniform-to-normal-distribution
    double u1 = randd();
    double u2 = randd();
    return mu + sqrt( -2.0 * log( u1 ) ) * cos( 2.0 * M_PI * u2 ) * sigma;
}



////////// FILE OPERATIONS /////////////////////////////////////////////////////////////////////////

bool file_exists( const string& fName ){ 
    // Return true if the file exists , otherwise return false
    struct stat buf; 
    if( stat( fName.c_str() , &buf ) != -1 ){ return true; } else { return false; }
}

vector<string> read_lines( string path ){ 
    // Return all the lines of text file as a string vector
    vector<string> rtnVec;
    if( file_exists( path ) ){
        ifstream fin( path ); // Open the list file for reading
        string   line; // String to store each line
        while ( std::getline( fin , line ) ){ // While we are able to fetch a line
            rtnVec.push_back( line ); // Add the file to the list to read
        }
        fin.close();
    } else { cout << "readlines: Could not open the file " << path << endl; }
    return rtnVec;
}

#endif // UTILS_HPP