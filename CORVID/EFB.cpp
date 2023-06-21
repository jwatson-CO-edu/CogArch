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



////////// EVOLUTIONARY FEATURE BUS ////////////////////////////////////////////////////////////////

enum EFB_Op{ 
    NOP, // NO-OP
    ADD, // Add
    SUB, // Subtract
    CON, // Constant
    DLY, // Delay
};

struct EFB_Feature{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // A single operation
    // 2023-06-20: At this time, each feature has a scalar output, but the idea of batch operations makes sense
    // 2023-06-21: Members could have probably been represented as matrices at the layer level, just MVP at this point please 

    /// Members ///
    EFB_Op /*------------*/ opType; // Type of operation
    vector<array<size_t,2>> addrs; //- Addresses for operand lookup
    double /*------------*/ param; //- Tuning parameter for the feature
    double /*------------*/ pAvg; // - Tuning parameter mean
    double /*------------*/ pVar; // - Tuning parameter variance
    double /*------------*/ lr; // --- Learning rate 
    uint /*--------------*/ bufLen; // Length of 
    deque<double> /*-----*/ vHst; // - History of values, used for Delay
    deque<double> /*-----*/ pHst; // - History of parameter values
    deque<double> /*-----*/ rHst; // - History of reward
    
    /// Constructor ///

    EFB_Feature(){
        // Default constructor
        opType = NOP;
        param  =  0.0;
        bufLen = 10;
        lr     = 0.001;
    }

    /// Methods ///

    void sample_parameter(){
        // Draw a parameter from a normal dist
        param = Box_Muller_normal_sample( pAvg, pVar );
    }

    void store_param_reward( double reward_t ){
        pHst.push_back( param    );
        rHst.push_back( reward_t );
        while( pHst.size() > bufLen ){
            pHst.pop_front();
            rHst.pop_front();
        }
    }

    double apply( const vd& operands ){
        // Apply the operation to the input operands to produce a scalar output signal
        double rtnRes = 0.0;
        uint   delay;
        switch( opType ){
            case ADD: // Add
                for( double operand : operands ){  rtnRes += operand;  }
                rtnRes += param;
                break;
            case SUB: // Subtract
                rtnRes = operands[0];
                for( size_t i = 1; i < operands.size(); i++ ){  rtnRes -= operands[i];  }
                rtnRes -= param;
                break;
            case CON: // Constant
                rtnRes = param;
                break;
            case DLY: //Delay
                delay  = (uint) max( 0.0, param );
                if( vHst.size() >= delay )  rtnRes = vHst[ delay-1 ];
                vHst.push_back( rtnRes );
                while( vHst.size() > delay ){
                    vHst.pop_front();
                }
                break;
            case NOP:
            default:
        }
        return rtnRes;
    }

    double param_grad_ascent(){
        // FIXME, START HERE: UNROLL DQs INTO MATRICES, CALC GRAD, AND ASCEND PARAMETER GRAD
    }

};

struct EFB_Layer{
    vd /*------------*/ y; // ------ Output
    vector<EFB_Feature> features; // Features that define the output
};

struct EFB{
    // Simplest Evolutionary Feature Bus (EFB)
    vd /*------------*/ x; // ------ Input
};