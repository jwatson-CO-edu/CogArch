/*
SOM.cpp
Self-Organizing Map Example
g++ SOM.cpp -std=gnu++17 -I /usr/include/eigen3
g++ SOM.cpp -std=gnu++17 -O3 -I /usr/include/eigen3
*/


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
    
    /// Constructor ///

    SelfOrgMapLayer( const MatrixXd& mapBounds, const vector<double>& resolution ){
        // Create a SOM layer with initial feature locations in a regular grid
        // 1. Infer input dimension and store bounds
        dI     = mapBounds.rows();
        bounds = mapBounds;
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
        // Find the closest point to the input vector, linear search
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

    void train_one_example( const vd& input ){
        // Perform the SOM training procedure for one example
        load_input( input );
        uint BMUdex = find_BMU_for_x();
        // FIXME: START HERE
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
    

    return 0;
}
