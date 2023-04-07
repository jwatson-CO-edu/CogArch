
////////// INIT ////////////////////////////////////////////////////////////////////////////////////

/// Standard Imports ///
#include <cmath>// ---- `exp` 
#include <stdlib.h> //- `srand`, `rand` 
#include <time.h> // -- `time` 
#include <algorithm> // `clamp`
using std::clamp;
#include <iostream>
using std::cout, std::endl, std::flush, std::ostream;
#include <vector>
using std::vector;
#include <set>
using std::set;
#include <string>
using std::string;
#include <fstream>
using std::ifstream, std::ofstream;
#include <sys/stat.h>
#include <filesystem>
#include <ctype.h> // `isspace`
#include <sstream>
using std::istringstream;

/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;



////////// HELPER FUNCTIONS ////////////////////////////////////////////////////////////////////////

template<typename T>
ostream& operator<<( ostream& os , const vector<T>& vec ) { 
    // ostream '<<' operator for vectors
    // NOTE: This function assumes that the ostream '<<' operator for T has already been defined
    os << "[ ";
    for (size_t i = 0; i < vec.size(); i++) {
        os << (T) vec[i];
        if (i + 1 < vec.size()) { os << ", "; }
    }
    os << " ]";
    return os; // You must return a reference to the stream!
}



////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////

float randf(){
    // Return a pseudo-random number between 0.0 and 1.0
    return  1.0f * rand() / RAND_MAX;
}

void seed_rand(){  srand( time( NULL ) );  } // Provide a clock-based unpredictable seed to RNG

float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }
float ddz_sigmoid( float sigZ ){  return sigZ * ( 1.0 - sigZ );  }


///// RANDOM SAMPLING ////////////////////////////

vector<float> sample_uniform_hypercube( size_t dim, float lo, float hi ){
    // Sample from an N-`dim`ensional hypercube with identical bounds for each dimension
    vector<float> rtnArr;
    float /*---*/ span = hi - lo;
    for( size_t i = 0; i < dim; i++ ){  rtnArr.push_back( lo + span*randf() );  }
    return rtnArr;
}