/*
SOM.cpp
Self-Organizing Map Example
g++ SOM.cpp -std=gnu++17 -I /usr/include/eigen3
g++ SOM.cpp -std=gnu++17 -O3 -I /usr/include/eigen3
*/

/// Standard ///
#include <map>
using std::pair;

/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;

/// Local ///
#include "utils.hpp"


////////// HELPER FUNCTIONS ////////////////////////////////////////////////////////////////////////


void populate_grid_points( MatrixXd& pntMatx, const vector<vector<double>>& tics ){
    // Populate `pntMatx` with rectangular grid points defined by `tics`
    int  Mrows  = 1;
    int  Ncols  = tics.size();
    uint repeat = 1;
    uint i,R,k,K;
    // Calculate grid size and allocate
    for( vector<double> dimTics : tics ){
        Mrows *= dimTics.size();
    }
    // cout << "Matrix will be " << Mrows << " x " << Ncols << endl;
    pntMatx = MatrixXd::Zero( Mrows, Ncols );
    // cout << "Matrix created!" << endl;
    
    // Repeat the tic the appropriate number of times for each dimension
    R = 1;
    // For each dimension
    for( int j = Ncols-1; j > -1; j-- ){
        // Reset the row counter
        i = k = 0;
        K = tics[j].size();
        while( i < Mrows ){
            for( uint r = 0; r < R; r++ ){
                // cout << "\t i= " << i << ", j= " << j
                pntMatx(i,j) = tics[j][k];      
                i++;  
            }
            k = (k+1)%K;
        }
        if( j > 0 ){
            R *= tics[j-1].size();
        }
    }
}

void random_elem_init_d( MatrixXd& W, double lo = 0.0f, double hi = 1.0f ){
    // Set all weights and biases to uniformly-distributed random numbers
    uint M = W.rows();
    uint N = W.cols();
    for( uint i = 0; i < M; i++ ){
        for( uint j = 0; j < N; j++ ){
            W(i,j) = randd( lo, hi );
        }
    }
}

inline string& ltrim_ws( string& s, const char* t = " \t\n\r\f\v" ){
    // trim whitespace from left
    s.erase( 0, s.find_first_not_of(t) );
    return s;
}


inline string& rtrim_ws( string& s, const char* t = " \t\n\r\f\v" ){
    // trim whitespace from right
    s.erase( s.find_last_not_of(t) + 1 );
    return s;
}


inline string& trim_ws( string& s, const char* t = " \t\n\r\f\v" ){
    // trim whitespace from left & right
    return ltrim_ws( rtrim_ws(s, t), t );
}

vstr split_string_sep( string input, char sepChar = ' ' ){
    // Return a vector of strings found in `input` separated by whitespace
    vstr   rtnWords;
    string currWord;
    char   currChar;
    size_t strLen = input.size();
    bool   wsMode = (sepChar == ' ');
    bool   p_sep  = false;

    input = trim_ws( input );
    input.push_back( sepChar ); // Separator hack
    
    for( size_t i = 0 ; i < strLen ; i++ ){
        // 1. Load char
        currChar = input[i];
        // 2. Check char
        if( wsMode && isspace( currChar ) )
            p_sep = true;
        else if( (!wsMode) && (currChar == sepChar) )
            p_sep = true;
        else
            p_sep = false;
        // 3. Handle char
        if( p_sep ){
            if( currWord.length() > 0 )  rtnWords.push_back( currWord );
            currWord = "";
        }else{
            currWord.push_back( currChar );
        }
    }
    return rtnWords; 
}

vd doublify_string_vec( const vstr& rawVec ){
    // Read a vector of strings into a vector of doubles
    vd rtnVec;
    for( string elem : rawVec ){  rtnVec.push_back( stod( elem ) );  }
    return rtnVec;
}





////////// CSV STRUCT //////////////////////////////////////////////////////////////////////////////

struct BufferCSVd{
    // Container for data read from CSV file, double data in ragged vector rows

    /// Members ///
    vvd data;

    /// Methods ///

    bool read_CSV( string fPath, char separator = ' ' ){
        // Read the contents of the CSV file into `data` as ragged vector rows of doubles
        vstr strRow;
        vd   dblRow;
        vstr rawLines = read_lines( fPath );
        uint i = 0;
        cout << rawLines[  0] << endl;
        cout << rawLines[100] << endl;
        for( string strLine : rawLines ){
            strRow = split_string_sep( strLine, separator );
            dblRow = doublify_string_vec( strRow );
            if( i < 2 ){
                cout << strRow << endl;
                cout << dblRow << endl;
            }
            data.push_back( dblRow );
            i++;
        }
        if(rawLines.size() > 0){
            cout << "Read in " << data.size() << " rows and " << data[0].size() << " columns!" << endl;
            return true;
        }else{
            cout << "No data in " << fPath << "!" << endl;
            return false;
        }
    }

    vvd get_column_ranges(){
        // Get the range of each dimension so a grid can be built
        // NOTE: This function assumes a uniform number of columns across rows, not ragged
        uint   rows = data.size();
        uint   cols = data[0].size();
        double datum;
        vvd    ranges;
        for( uint i = 0; i < cols; i++ ){  ranges.push_back( { 1e9,-1e9} );  }
        for( uint i = 0; i < rows; i++ ){
            for( uint j = 0; j < cols; j++ ){
                datum = data[i][j];
                if( datum < ranges[j][0] )  ranges[j][0] = datum;
                if( datum > ranges[j][1] )  ranges[j][1] = datum;
            }
        }
        return ranges;
    }
    
};



////////// SELF-ORGANIZING MAP /////////////////////////////////////////////////////////////////////

struct SelfOrgMapLayer{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Self-Organizing Map

    /// Members ///
    uint     dI; // ---- Dimensionality of the input 
    uint     Nout; // -- Number of elements in the feature map (output) 
    MatrixXd W; // ----- Weight matrix relating input to features
    MatrixXd x;
    MatrixXd m;
    MatrixXd bounds; //- Hypercuboid defining the bounds of the map
    MatrixXd maplocs; // Locations of feature map elements
    double   releRad; // Relevance radius
    double   scale; // - Problem scale
    double   lr; // ---- Learning rate
    
    /// Constructor ///

    SelfOrgMapLayer( const MatrixXd& mapBounds, const vector<double>& resolution, double learnRate, 
                     double searchRadius, double problemScale = 1.0 ){
        // Create a SOM layer with initial feature locations in a regular grid
        // 1. Infer input dimension and store params
        dI      = mapBounds.rows();
        bounds  = mapBounds;
        releRad = searchRadius; // FIXME: SHOULD THIS DECAY, AND IF SO, HOW?
        scale   = problemScale;
        lr /**/ = learnRate;
        // 2. Calculate the gradations for each of the dimensions and store in a ragged double vector
        vector<vector<double>> tics;
        vector<double> /*---*/ dimTics;
        double val, lim, res;
        for( uint i = 0; i < dI; i++ ){
            val = mapBounds(i,0);
            lim = mapBounds(i,1);
            res = resolution[i];
            dimTics.clear();
            dimTics.push_back( val );
            val += res;
            while( val < lim ){
                dimTics.push_back( val );
                val += res;
            }
            tics.push_back( dimTics );
        }
        populate_grid_points( maplocs, tics );
        Nout = maplocs.rows();
        W    = MatrixXd::Zero( Nout, dI );
        x    = MatrixXd::Zero( 1   , dI );
        random_elem_init_d( W, 0.01, 0.75 );
    }

    /// Methods ///

    void load_input( const vd& x_t ){
        // Load values into the input vector
        // NOTE: This struct does not require the client code to add the unity input bias
        for( uint i = 0; i < dI; i++ ){  x(0,i) = x_t[i];  }
    }

    uint find_BMU_for_x(){
        // Find the closest weight vector to the input vector, linear search
        MatrixXd diff;
        double   dist;
        double   dMin = 1e9;
        double   iMin = 0;
        for( uint i = 0; i < Nout; i++ ){
            diff = W.block(i,0,1,dI) - x;
            dist = diff.norm();
            if( dist < dMin ){
                dMin = dist;
                iMin = i;
            }
        }
        return iMin;
    }

    vector<pair<uint,double>> get_BMU_neighborhood( uint BMUdex, double radius ){
        double /*--------------*/ dist;
        MatrixXd /*------------*/ diff;
        MatrixXd /*------------*/ BMU = maplocs.block( BMUdex, 0, 1, dI );
        vector<pair<uint,double>> rtnDices;
        for( uint i = 0; i < Nout; i++ ){
            diff = maplocs.block(i,0,1,dI) - x;
            dist = diff.norm();
            // if( (dist <= radius) && (dist > 0.0) ){
            if( dist <= radius ){
                // Membership is determined by an exponential decay function
                rtnDices.push_back( pair<uint,double>{i, exp(-dist/scale)} );
            }
        }
        return rtnDices;
    }

    void train_one_example( const vd& input ){
        // Perform the SOM training procedure for one example
        double alpha;
        uint   index;
        // 0. Store input
        load_input( input );
        // 1. Compute the Best Matching Unit and its neighborhood
        uint /*----------------*/ BMUdex /*-*/ = find_BMU_for_x();
        vector<pair<uint,double>> neighborhood = get_BMU_neighborhood( BMUdex, releRad );
        // 2. Each neighboring node’s  weights are adjusted to make them more like the input vector
        for( pair<uint,double> elem : neighborhood ){
            // `alpha`: The closer a node is to the BMU; the more its weights get altered.
            alpha = lr * elem.second;
            index = elem.first;
            W.block( index, 0, 1, dI ) = (1.0-alpha) * W.block( index, 0, 1, dI ) + alpha * x;
        }
    }
};



////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

int main(){

    ///// Test 0: Grid Point Creation /////

    if( false ){
        MatrixXd /*---------*/ combos;
        vector<vector<double>> tics = {{1,2,3},{1,2,3},{1,2,3}};
        populate_grid_points( combos, tics );
        cout << combos << endl;
    }

    ///// Test 1: Read CSV /////
    if( true ){
        BufferCSVd buf;
        buf.read_CSV( "../Data/Seizure-Data/seizure-reduced-10col_no-headings.csv", ',' );
        cout << buf.get_column_ranges() << endl;
    }
    

    return 0;
}
