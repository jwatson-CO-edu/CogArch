/*
SOM.cpp
Self-Organizing Map Example
g++ SOM.cpp -std=gnu++17 -O3 -I /usr/include/eigen3
WARNING: This implementation is wildly unoptimized and suffers badly from a lack of spatial tree lookup!
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

vvd get_N_tics( const vui& Ntics, const vvd& ranges ){
    // Separate each dimension's range into per-dimension `Ntics`
    // NOTE: This function assumes that `Ntics` and `ranges` are the same length
    uint   Ndim = Ntics.size();
    uint   N_i;
    double span_i, div, val;
    vd     dimTics;
    vvd    tics;
    
    for( uint i = 0; i < Ndim; i++ ){
        span_i = ranges[i][1] - ranges[i][0];
        N_i    = Ntics[i]-1;
        div    = span_i / (1.0 * N_i);
        val    = ranges[i][0];
        dimTics.clear();
        dimTics.push_back( val );
        for( uint j = 0; j < N_i; j++ ){
            val += div;
            dimTics.push_back( val );
        }
        tics.push_back( dimTics );
    }
    return tics;
}


void populate_grid_points( MatrixXd& pntMatx, const vvd& tics ){
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
    // Original Author: Galik, https://stackoverflow.com/a/25829233
    s.erase( 0, s.find_first_not_of(t) );
    return s;
}


inline string& rtrim_ws( string& s, const char* t = " \t\n\r\f\v" ){
    // trim whitespace from right
    // Original Author: Galik, https://stackoverflow.com/a/25829233
    s.erase( s.find_last_not_of(t) + 1 );
    return s;
}


inline string& trim_ws( string& s, const char* t = " \t\n\r\f\v" ){
    // trim whitespace from left & right
    // Original Author: Galik, https://stackoverflow.com/a/25829233
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
    vvd    data;
    size_t rowDex;
    size_t Mrows;
    size_t Ncols;

    /// Constructor ///

    BufferCSVd( string fPath, char separator = ' ' ){
        // Set the current data row and read the file by rows
        rowDex = 0;
        read_CSV( fPath, separator );
        Mrows = data.size();
        Ncols = data[0].size();
    }

    /// Methods ///

    vd get_next_row(){
        // Fetch row and advance the row index
        vd rtnRow = data[ rowDex ];
        rowDex = (rowDex+1) % data.size();
        return rtnRow;
    }

    bool read_CSV( string fPath, char separator = ' ' ){
        // Read the contents of the CSV file into `data` as ragged vector rows of doubles
        vstr strRow;
        vd   dblRow;
        vstr rawLines = read_lines( fPath );
        uint i = 0;
        // cout << rawLines[  0] << endl;
        // cout << rawLines[100] << endl;
        for( string strLine : rawLines ){
            strRow = split_string_sep( strLine, separator );
            dblRow = doublify_string_vec( strRow );
            // if( i < 2 ){
            //     cout << strRow << endl;
            //     cout << dblRow << endl;
            // }
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
        cout << ranges << endl;
        return ranges;
    }
    
    vvd even_tics_for_grid_covering_data( uint Ntics ){
        // Separate each dimension of the dataset into `Ntics` for a grid that covers the dataset domain
        vvd colRanges = get_column_ranges();
        vui NperDim;
        for( uint i = 0; i < colRanges.size(); i++ ){  NperDim.push_back( Ntics );  }
        return get_N_tics( NperDim, colRanges ); 
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
    MatrixXd maplocs; // Locations of feature map elements
    double   releRad; // Relevance radius
    double   scale; // - Problem scale
    double   lr; // ---- Learning rate
    double   dc; // ---- Decay constant
    
    /// Constructor & Init ///

    SelfOrgMapLayer( double learnRate, double searchRadius, double problemScale = 1.0, double decayConst = 0.99 ){
        // Create a SOM layer with initial feature locations in a regular grid
        // 1. Infer input dimension and store params
        releRad = searchRadius; // FIXME: SHOULD THIS DECAY, AND IF SO, HOW?
        dc /**/ = decayConst;
        scale   = problemScale;
        lr /**/ = learnRate;
    }

    void structure_init_from_tics( const vvd& tics ){
        // Populate the grid according to per-dimension `tics` provided by the `BufferCSVd`
        populate_grid_points( maplocs, tics );
        dI   = tics.size();
        Nout = maplocs.rows();
        W    = MatrixXd::Zero( Nout, dI );
        x    = MatrixXd::Zero( 1   , dI );
        // random_elem_init_d( W, 0.01, 0.75 );
        random_elem_init_d( W, -2000.0, 2000.0 );
    }

    /// Methods ///

    void load_input( const vd& x_t ){
        // Load values into the input vector
        // NOTE: This struct does not require the client code to add the unity input bias
        for( uint i = 0; i < dI; i++ ){  x(0,i) = x_t[i];  }
    }

    uint find_BMU_for_x( bool report = false ){
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
        if( report )  cout << "Example: " << x << endl << "BMU Weights: " << W.block(iMin,0,1,dI) << endl;
        return iMin;
    }

    vector<pair<uint,double>> get_BMU_neighborhood( uint BMUdex, double radius, bool report = false ){
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
        if( report )  cout << "There are " << rtnDices.size() << " neighbors of the BMU, " << BMU << endl;
        return rtnDices;
    }

    double train_one_example( const vd& input, bool report = false ){
        // Perform the SOM training procedure for one example
        double   alpha;
        uint     index;
        MatrixXd Wrow;
        double   delta = 0.0;
        // 0. Store input
        load_input( input );
        // 1. Compute the Best Matching Unit and its neighborhood
        uint /*----------------*/ BMUdex /*-*/ = find_BMU_for_x( report );
        vector<pair<uint,double>> neighborhood = get_BMU_neighborhood( BMUdex, releRad, report );
        // 2. Each neighboring node’s  weights are adjusted to make them more like the input vector
        for( pair<uint,double> elem : neighborhood ){
            // `alpha`: The closer a node is to the BMU; the more its weights get altered.
            alpha = lr * elem.second;
            index = elem.first;
            Wrow  = W.block( index, 0, 1, dI );
            W.block( index, 0, 1, dI ) = (1.0-alpha) * Wrow + alpha * x;
            if( report )  delta += (Wrow - W.block( index, 0, 1, dI )).norm();
        }
        if( report )  cout << "Net weights changes: " << delta << endl << endl;
        // 3. Search radius decays exponentially
        releRad *= dc;
        // N. Return dist from BMU weights to eval convergence
        return (x - W.block( BMUdex, 0, 1, dI )).norm();
    }
};

void train_one_epoch( BufferCSVd& buf, SelfOrgMapLayer& som, size_t reportDiv = 100 ){
    // Train on every example of the CSV buffer
    bool   report = false;
    double dTotal = 0.0;
    for( size_t i = 0; i < buf.Mrows; i++ ){
        report = (i%reportDiv==0);
        if( report )  cout << endl << endl;
        dTotal += som.train_one_example( buf.get_next_row(), report );    
        if( report ){
            cout << "Average distance to BMU: " << (dTotal/(1.0*reportDiv)) << endl;
            dTotal = 0.0;
            cout << endl;
        }else
            cout << "." << flush;
    }
}

double evaluate_on_CSV( BufferCSVd& buf, SelfOrgMapLayer& som, size_t reportDiv = 100 ){
    // Reset the CSV, then evaluate the average dist between each datum and its assigned cluster
    bool   report = false;
    double dTotal = 0.0;
    uint   BMUdex;
    buf.rowDex = 0;
    for( size_t i = 0; i < buf.Mrows; i++ ){
        report = (i%reportDiv==0);
        som.load_input( buf.get_next_row() );
        // 1. Compute the Best Matching Unit
        BMUdex = som.find_BMU_for_x( report );
        dTotal += (som.x - som.W.block( BMUdex, 0, 1, som.dI )).norm();    
        if( report )  cout << ">" << flush;
    }
    cout << endl << endl << "########## Self-Organizing Map Report ##########" << endl;
    cout << "Average distance to BMU: " << (dTotal/(1.0*buf.Mrows)) << endl;
    cout << "Final Search Radius: ___ " << som.releRad << endl;
    cout << "Learning Rate: _________ " << som.lr << endl;
    cout << "Problem Scale: _________ " << som.scale << endl;
    cout << "Decay Constant: ________ " << som.dc << endl << endl;
}

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
    if( false ){
        BufferCSVd buf{ "../Data/Seizure-Data/seizure-reduced-10col_no-headings.csv", ',' };
        cout << buf.get_column_ranges() << endl;
    }

    ///// Test 2: Grid from CSV, Teach one row /////
    // 5^10 =  9,765,625
    // 6^10 = 60,466,176
    if( false ){
        BufferCSVd buf{ "../Data/Seizure-Data/seizure-reduced-10col_no-headings.csv", ',' };
        SelfOrgMapLayer som{ 0.001, 4000.0/5.0*1.25, 4000.0/5.0 };
        cout << "About to build structure ..." << endl;
        som.structure_init_from_tics( buf.even_tics_for_grid_covering_data( 5 ) );
        cout << som.W.rows() << " x " << som.W.cols() << endl; // 9765625, Correct!
        som.train_one_example( buf.get_next_row() );
    }

    ///// Test 3: Train one epoch /////
    if( true ){
        BufferCSVd /**/ buf{ "../Data/Seizure-Data/seizure-reduced-10col_no-headings.csv", ',' };
        SelfOrgMapLayer som{ 0.02, 4000.0/2.0, 4000.0/5.0, 0.9999 };
        cout << "About to build structure ..." << endl;
        som.structure_init_from_tics( buf.even_tics_for_grid_covering_data( 5 ) );
        cout << som.W.rows() << " x " << som.W.cols() << endl; // 9765625, Correct!
        train_one_epoch( buf, som, 100 );
    }

    return 0;
}
