
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
    uint i;
    
    // Calculate grid size and allocate
    for( vector<double> dimTics : tics ){
        Mrows *= dimTics.size();
    }
    pntMatx = MatrixXd::Zero( Mrows, Ncols );
    
    // Repeat the tic the appropriate number of times for each dimension
    // For each dimension
    for( int j = Ncols-1; j > -1; j-- ){
        // Reset the row counter
        i = 0;

        // FIXME, START HERE: DO WHAT IT SAYS YOU WAS GONNA DO

    }
}



////////// SELF-ORGANIZING MAP /////////////////////////////////////////////////////////////////////

struct SelfOrgMapLayer{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Self-Organizing Map

    /// Members ///
    uint     dI; // ---- Dimensionality of the input 
    uint     Nout; // -- Number of elements in the feature map (output) 
    MatrixXd W; // ----- Weight matrix relating input to features
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
    }

    /// Methods ///

};