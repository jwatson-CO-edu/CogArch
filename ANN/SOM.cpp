
/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;

/// Local ///
#include "utils.hpp"

struct SelfOrgMapLayer{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Self-Organizing Map

    /// Members ///
    uint     dI; // ---- Dimensionality of the input 
    uint     Nout; // -- Number of elements in the feature map (output) 
    MatrixXd W; // ----- Weight matrix relating input to features
    MatrixXd bounds; //- Hypercuboid defining the bounds of the map
    MatrixXd maplocs; // Locations of feature map elements
    
    /// Constructor ///

    SelfOrgMapLayer( const MatrixXd& mapBounds, const MatrixXd& resolution ){
        // Create a SOM layer with initial feature locations in a regular grid
        // 1. Infer input dimension and store bounds
        dI     = mapBounds.rows();
        bounds = mapBounds;
        // 2. FIXME, START HERE: CALCULATE GRID POINTS AND DETERMINE NUMBER OF FEATURES
        vector<vector<uint>>
    }

    /// Methods ///

};