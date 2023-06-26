/*
EFB.cpp
Evolutionary Feature Bus (EFB)
g++ EFB.cpp -std=gnu++17 -O3 -I /usr/include/eigen3
g++ EFB.cpp -std=gnu++17 -O3 -I ../include -I /usr/include/eigen3
EFB is an experimental attempt to use Genetic Programming for automated feature engineering
This program is to test if EFB can represent an arbitrary function.
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

///// Imports ////////////////////////////////////

/// Standard ///
#include <deque>
using std::deque;

/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;

/// Local ///
// #include "../include/utils.hpp"
#include "utils.hpp"


///// Aliases ////////////////////////////////////
typedef Eigen::MatrixXd matxX;

////////// EVOLUTIONARY FEATURE BUS ////////////////////////////////////////////////////////////////

///// Operation Types ////////////////////////////

enum EFB_Op{ 
    // Possible operation types
    NOP = 0, // NO-OP
    ADD = 1, // Add
    SUB = 2, // Subtract
    CON = 3, // Constant
    DLY = 4, // Delay
};

///// Individual Operation ///////////////////////

struct EFB_Feature{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // A single operation
    // 2023-06-20: At this time, each feature has a scalar output, but the idea of batch operations makes sense
    // 2023-06-21: Members could have probably been represented as matrices at the layer level, just MVP at this point please 

    /// Members ///
    EFB_Op /*----------*/ opType; // Type of operation
    vector<array<uint,2>> addrs; //- Addresses for operand lookup
    double /*----------*/ param; //- Tuning parameter for the feature
    double /*----------*/ pAvg; // - Tuning parameter mean
    double /*----------*/ pVar; // - Tuning parameter variance
    double /*----------*/ lr; // --- Learning rate 
    double /*----------*/ output; // Output 
    uint /*------------*/ Nhst; // - Steps of history to retain
    deque<double> /*---*/ vHst; // - History of values, used for Delay
    deque<double> /*---*/ pHst; // - History of parameter values
    deque<double> /*---*/ eHst; // - History of error
    
    /// Constructor ///

    EFB_Feature(){
        // Default constructor
        opType = NOP;
        param  =  0.0;
        Nhst   = 10;
        lr     =  0.001;
    }

    /// Methods ///

    void sample_parameter(){
        // Draw a parameter from a normal dist
        param = Box_Muller_normal_sample( pAvg, pVar );
    }

    void store_step_values( double error_t ){
        // Store the parameter and the error associated with it
        vHst.push_front( output  );
        pHst.push_front( param   );
        eHst.push_front( error_t );
        while( pHst.size() > Nhst ){
            vHst.pop_back();
            pHst.pop_back();
            eHst.pop_back();
        }
    }

    double apply( const vd& operands ){
        // Apply the operation to the input operands to produce a scalar output signal
        output = 0.0;
        uint delay;
        switch( opType ){
            case ADD: // Add
                for( double operand : operands ){  output += operand;  }
                output += param;
                break;
            case SUB: // Subtract
                output = operands[0];
                for( size_t i = 1; i < operands.size(); i++ ){  output -= operands[i];  }
                output -= param;
                break;
            case CON: // Constant
                output = param;
                break;
            case DLY: //Delay
                delay  = (uint) max( 0.0, param );
                if( vHst.size() ){
                    if( vHst.size() >= delay ){
                        output = vHst[ delay-1 ];
                    }else{
                        output = vHst.back();
                    }
                }else{
                    output = operands[0];
                }
                break;
            case NOP:
            default:
                output = 0.0;
        }
        return output;
    }

    double param_grad_descent(){
        // Calc gradient and adjust the parameter
        uint  Nelem = pHst.size();
        matxX P     = matxX::Ones( Nelem, 2 );
        matxX E     = matxX::Ones( Nelem, 1 );
        for( uint i = 0; i < Nelem; i++ ){
            P(i,0) = pHst[i];
            E(i,0) = eHst[i];
        }
        matxX  soln = (P.transpose() * P).ldlt().solve(P.transpose() * E);
        double grad = soln(0,0);
        pAvg -= grad * lr;
        return grad;
    }

    matxX get_history(){
        // Get the history of each of the queues
        // NOTE: ROWS OF MATRIX IN REVERSE TIME ORDER
        matxX rtnArr = matxX::Zero( Nhst, 3 );
        for( uint i = 0; i < Nhst; i++ ){
            if( i < vHst.size() ){
                rtnArr(i,0) = vHst[i];
                rtnArr(i,1) = pHst[i];
                rtnArr(i,2) = eHst[i];
            }
        }
        return rtnArr;
    }
};

///// Layer Struct ///////////////////////////////

struct EFB_Layer{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // A layer of transformed input

    /// Members ///
    vector<EFB_Feature> features; // Features that define the output
    uint /*----------*/ Nhst; // ----- Steps of history to retain
    vd /*------------*/ y; // ------ Output

    /// Methods ///

    matxX get_history(){
        // Get the history of all features 
        matxX rtnArr = matxX::Zero( Nhst, 3*features.size() );
        for( uint i = 0; i < Nhst; i++ ){
            rtnArr.block( 0, i*3, Nhst, 3 ) = features[i].get_history();
        }
        return rtnArr;
    }

};

///// Evolutionary Feature Bus ///////////////////

struct EFB{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Evolutionary Feature Bus (EFB)
    uint /*------------*/ dI; // ------- Input dimension 
    vd /*--------------*/ x; // -------- Input vector
    uint /*------------*/ operMax; // -- Max number of operands for any operation
    uint /*------------*/ dpthMax; // -- Max depth beyond input
    uint /*------------*/ Nhst; // ----- Steps of history to retain
    vector<EFB_Layer>     layers; // --- Layers that define the bus
    vector<array<uint,2>> addrs; // ---- Available addresses
    array<double,2> /*-*/ paramRange; // Range that a parameter can take
    double /*----------*/ paramVar; // - Variance of parameters

    /// Constructor ///

    EFB( uint inputDim, uint maxOperands, uint maxDepth ){
        // Init input
        dI = inputDim;
        // The input is Layer 0
        for( uint i = 0; i < dI; i++ ){  addrs.push_back( {0, i} );  }
        operMax = maxOperands;
        dpthMax = maxDepth;
    }

    /// Methods ///

    void load_input( const vd& input ){
        // Copy the input and return the length of the input
        x.clear();
        x = input;
        if( x.size() != dI )  cout << "Input dimension " << x.size() << ", but expected " << dI << " !" << endl;
    }

    bool create_feature(){
        // Create a new feature, assuming the decision to create the feature has already been made
        // 0. Init
        EFB_Feature rtnFeature;
        bool /*--*/ placed = false;
        // 1. Choose the number of operands
        uint Noper = randu( 1, min( operMax, (uint) addrs.size() ) );
        uint Naddr = addrs.size();
        // 2. Choose the addresses
        vector<array<uint,2>> opAddrs;
        array<uint,2> /*---*/ oneAddr;
        for( uint i = 0; i < Noper; i++ ){
            oneAddr = addrs[ randu(0, Naddr-1) ];
            while( is_arg_in_vec( oneAddr, opAddrs ) ){  oneAddr = addrs[ randu(0, Naddr-1) ];  }
            opAddrs.push_back( oneAddr );
        }
        rtnFeature.addrs = opAddrs;
        // 3. Choose the operation type
        rtnFeature.opType = static_cast<EFB_Op>( randu(1, 4) );
        // 4. Choose the parameter 
        rtnFeature.pAvg = randd( paramRange[0], paramRange[1] );
        rtnFeature.pVar = paramVar;
        // 5. Place the feature, Report placement
        uint maxLvl = 0;
        uint nxtLvl;
        for( array<uint,2> addr : opAddrs ){
            if( addr[0] > maxLvl )  maxLvl = addr[0];
        }
        nxtLvl = maxLvl+1;
        if(nxtLvl > dpthMax){
            return false;
        }else{
            if(nxtLvl > layers.size())  layers.push_back( EFB_Layer{} );
            layers[ nxtLvl-1 ].features.push_back( rtnFeature );
            layers[ nxtLvl-1 ].y.push_back( 0.0 );
            return true;
        }
    }

    bool create_feature( const vector<array<uint,2>>& opAddrs, EFB_Op typ, double paramAvg, double pVariance ){
        // Create a new feature, assuming the decision to create the feature has already been made
        // 0. Init
        EFB_Feature rtnFeature;
        bool /*--*/ placed = false;
        // 1. Choose the number of operands
        uint Noper = randu( 1, min( operMax, (uint) addrs.size() ) );
        uint Naddr = addrs.size();
        // 2. Choose the addresses
        rtnFeature.addrs = opAddrs;
        // 3. Choose the operation type
        rtnFeature.opType = typ;
        // 4. Choose the parameter 
        rtnFeature.pAvg = paramAvg;
        rtnFeature.pVar = pVariance;
        // 5. Place the feature, Report placement
        uint maxLvl = 0;
        uint nxtLvl;
        for( array<uint,2> addr : opAddrs ){
            if( addr[0] > maxLvl )  maxLvl = addr[0];
        }
        nxtLvl = maxLvl+1;
        if(nxtLvl > dpthMax){
            return false;
        }else{
            if(nxtLvl > layers.size())  layers.push_back( EFB_Layer{} );
            layers[ nxtLvl-1 ].features.push_back( rtnFeature );
            layers[ nxtLvl-1 ].y.push_back( 0.0 );
            return true;
        }
    }

    vd fetch_input( const vector<array<uint,2>>& addrs ){
        // Get the necessary inputs to this feature
        // NOTE: Layer 0 is the EFB input
        vd   rtnArr;
        for( array<uint,2> addr_i : addrs ){
            if( addr_i[0] == 0 ){
                rtnArr.push_back( x[ addr_i[1] ] );
            }else{
                rtnArr.push_back( layers[ addr_i[0]-1 ].y[ addr_i[1] ] );
            }
        }
        return rtnArr;
    }

    void publish_output(){
        // Move output from features to output arrays
        for( EFB_Layer layer : layers ){
            uint i = 0;
            for( EFB_Feature feature : layer.features ){
                layer.y[i] = feature.output;
                i++;
            }
        }
    }

    void apply(){
        // Run all features
        for( EFB_Layer layer : layers ){
            for( EFB_Feature feature : layer.features ){
                feature.apply( fetch_input( feature.addrs ) );
            }
        }
    }

    vd get_output(){
        // Return the output of the last layer
        return layers.back().y;
    }
};

////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

int main(){

    ///// Test 1: Manually Add One Feature /////
    if( true ){
        EFB efb{ 1, 10, 5 };
        vector<array<uint,2>> inAddr;
        inAddr.push_back( {0,0} );
        efb.create_feature( inAddr, ADD, 5.0, 0.5 );
        cout << "There are " << efb.layers.size() << " layers!" << endl;
        efb.load_input( {2.0} );
        efb.apply();
        efb.publish_output();
        cout << "EFB output: " << efb.get_output() << endl;
    }

    return 0;
}
