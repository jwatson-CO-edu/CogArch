/*
EFB.cpp
Evolutionary Feature Bus (EFB)
g++ EFB.cpp -std=gnu++17 -O3 -I /usr/include/eigen3
g++ EFB.cpp -std=gnu++17 -O3 -I ../include
EFB is an experimental attempt to use Genetic Programming for automated feature engineering
This program is to test if EFB can represent an arbitrary function.
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

///// Imports ////////////////////////////////////

/// Standard ///
#include <deque>
using std::deque;

/// Local ///
#include "utils.hpp"



////////// EVOLUTIONARY FEATURE BUS ////////////////////////////////////////////////////////////////

enum EFB_Op{ 
    NOP, // NO-OP
    ADD, // Add
    SUB, // Subtract
    CON, // Constant
    DLY, // Delay
};

struct EFB_Feature{
    // A single operation
    // 2023-06-20: At this time, each feature has a scalar output, but the idea of batch operations makes sense

    /// Members ///
    EFB_Op /*------------*/ opType; // Type of operation
    vector<array<size_t,2>> addrs; //- Addresses for operand lookup
    double /*------------*/ param; //- Tuning parameter for the feature
    deque<double> /*-----*/ hist; // - History of values, used for Delay
    
    /// Constructor ///

    EFB_Feature(){
        param  = 0.0;
        opType = NOP;
    }

    /// Methods ///

    double apply( const vd& operands ){
        double rtnRes = 0.0;
        switch( opType ){
            case ADD:
                for( double operand : operands ){  rtnRes += operand;  }
                rtnRes += param;
                break;
            case SUB:
                rtnRes = operands[0];
                for( size_t i = 1; i < operands.size(); i++ ){  rtnRes -= operands[i];  }
                rtnRes -= param;
                break;
            case CON:
                rtnRes = param;
                break;
            case DLY:
                // FIXME, START HERE: APPLY A DELAY OF LENGTH `(uint) param`
                break;
            case NOP:
            default:
        }
        return rtnRes;
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