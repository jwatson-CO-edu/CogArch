/*
EFB.cpp
Evolutionary Feature Bus (EFB)
g++ EFB.cpp -std=gnu++17 -O3 -I ../include -I /usr/include/eigen3
EFB is an experimental attempt to use Genetic Programming for automated feature engineering
This program is to test if EFB can represent an arbitrary function.
WARNING: MASSIVE INEFFICIENCIES, EXPERIMENT ONLY
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
typedef array<uint,2>   address;

////////// HELPER FUNCTIONS ////////////////////////////////////////////////////////////////////////

double precent_difference( double op1, double op2 ){
    // Calc the Percent Difference according to https://www.mathsisfun.com/percentage-difference.html
    return abs( op1 - op2 ) / ( (op1+op2)/2.0 ) * 100.0;
}

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
    EFB_Op /*----*/ opType; // Type of operation
    vector<address> addrs; //- Addresses for operand lookup
    double /*----*/ param; //- Tuning parameter for the feature
    double /*----*/ pAvg; // - Tuning parameter mean
    double /*----*/ pVar; // - Tuning parameter variance
    double /*----*/ lr; // --- Learning rate 
    double /*----*/ output; // Output 
    uint /*------*/ Nhst; // - Steps of history to retain
    deque<double>   vHst; // - History of values, used for Delay
    double /*----*/ score; //- Fitness value of this feature
    
    /// Constructor ///

    EFB_Feature(){
        // Default constructor
        opType = NOP;
        param  =  0.0;
        Nhst   = 10;
        lr     =  0.001;
        score  = -1e9;
    }

    /// Methods ///

    void sample_parameter(){
        // Draw a parameter from a normal dist
        param = Box_Muller_normal_sample( pAvg, pVar );
    }

    void store_step_values(){
        // Store the parameter and the error associated with it
        vHst.push_front( output );
        while( vHst.size() > Nhst ){
            vHst.pop_back();
        }
    }

    double apply( const vd& operands ){
        // Apply the operation to the input operands to produce a scalar output signal
        output = 0.0;
        uint delay;
        switch( opType ){
            case ADD: // Add
                cout << "Apply ADD ..." << endl;
                for( double operand : operands ){  output += operand;  }
                output += param;
                break;
            case SUB: // Subtract
                cout << "Apply SUB ..." << endl;
                output = operands[0];
                for( size_t i = 1; i < operands.size(); i++ ){  output -= operands[i];  }
                output -= param;
                break;
            case CON: // Constant
                cout << "Apply CON ..." << endl;
                output = param;
                break;
            case DLY: //Delay
                cout << "Apply DLY ..." << endl;
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
                cout << "Apply NOP ..." << endl;
            default:
                cout << "Apply `default` ..." << endl;
                // output = 0.0;
        }
        cout << "Feature Output: " << output << endl;
        return output;
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

    

};

///// Evolutionary Feature Bus ///////////////////

struct EFB{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Evolutionary Feature Bus (EFB)
    uint /*--------*/ dI; // ------- Input dimension 
    vd /*----------*/ x; // -------- Input vector
    uint /*--------*/ operMax; // -- Max number of operands for any operation
    uint /*--------*/ dpthMax; // -- Max depth beyond input
    uint /*--------*/ Nhst; // ----- Steps of history to retain
    vector<EFB_Layer> layers; // --- Layers that define the bus
    vector<address>   addrs; // ---- Available addresses
    array<double,2>   paramRange; // Range that a parameter can take
    double /*------*/ paramVar; // - Variance of parameters
    deque<vd> /*---*/ vHst; // ----- History of values, used for Delay
    deque<vd> /*---*/ pHst; // ----- History of parameter values
    deque<vd> /*---*/ lHst; // ----- History of loss

    /// Constructor ///

    EFB( uint inputDim, uint maxOperands, uint maxDepth, const array<double,2>& pRange ){
        // Init input
        dI = inputDim;
        // The input is Layer 0
        for( uint i = 0; i < dI; i++ ){  addrs.push_back( {0, i} );  }
        operMax    = maxOperands;
        dpthMax    = maxDepth;
        paramRange = pRange;
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
        vector<address> opAddrs;
        address /*---*/ oneAddr;
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

    bool create_feature( const vector<address>& opAddrs, EFB_Op typ, double paramAvg, double pVariance ){
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

    vd fetch_input( const vector<address>& addrs ){
        // Get the necessary inputs to this feature
        // NOTE: Layer 0 is the EFB input
        vd   rtnArr;
        cout << "Retrieve Operands: " << addrs << endl;
        for( array<uint,2> addr_i : addrs ){
            if( addr_i[0] == 0 ){
                rtnArr.push_back( x[ addr_i[1] ] );
            }else{
                rtnArr.push_back( layers[ addr_i[0]-1 ].y[ addr_i[1] ] );
            }
        }
        cout << "Input Obtained!: " << rtnArr << endl;
        return rtnArr;
    }

    void sample_parameters(){
        // Run all features
        for( EFB_Layer& layer : layers ){
            for( EFB_Feature& feature : layer.features ){
                feature.sample_parameter();
            }
        }
    }

    void publish_output(){
        // Move output from features to output arrays
        for( EFB_Layer& layer : layers ){
            uint i = 0;
            for( EFB_Feature& feature : layer.features ){
                layer.y[i] = feature.output;
                i++;
            }
            cout << "Published: " << layer.y << endl;
        }
    }

    void apply(){
        // Run all features
        // sample_parameters();
        for( EFB_Layer& layer : layers ){
            for( EFB_Feature& feature : layer.features ){
                feature.output = feature.apply( fetch_input( feature.addrs ) );
                cout << "Feature stored: " << feature.output << endl;
            }
        }
    }

    vd get_output(){
        // Return the output of the last layer
        return layers.back().y;
    }

    vd forward( const vd& input ){
        // Run one input --to-> output cycle and return
        load_input( input );
        sample_parameters();
        apply();
        publish_output();
        return get_output();
    }

    vd get_output_params(){
        // Get the parameters of the output layer for this iteration
        vd rtnParams;
        for( EFB_Feature& feature : layers.back().features ){  rtnParams.push_back( feature.output );  }  
        return rtnParams;
    }

    void store_iter( const vd& loss ){
        // Store info for this iteration
        // NOTE: This function assumes that `sample_parameters` and `publish_output` have already been run this iteration
        const EFB_Layer& lastLayer = layers.back();
        vHst.push_front( get_output() );
        lHst.push_front( loss );
        pHst.push_front( get_output_params() );
    }

    vd score_output_and_calc_loss( double signalTs ){
        // Given the ground truth `signalTs`, Score each of the outputs and the features they depend on
        // 0. Init
        vd /*-------*/ outLoss;
        double /*---*/ loss_i, score_i;
        uint /*-----*/ i = 0;
        deque<address> frontier;
        address /*--*/ addr_j;
        // 1. For each output feature, compare to the actual
        for( double Yfeat : layers.back().y ){
            // 2. Calculate loss and score
            loss_i  = signalTs - Yfeat; // Actual - Predicted
            outLoss.push_back( loss_i );
            score_i = 100.0 - precent_difference( signalTs, Yfeat );
            // 3. Update parents
            frontier.push_back( {(uint)layers.size()-1, (uint)i} );
            while( frontier.size() ){
                addr_j = frontier.front();
                frontier.pop_front();
                if( addr_j[0] > 0 ){
                    EFB_Feature& currFeat = layers[ addr_j[0]-1 ].features[ addr_j[1] ];
                    if( currFeat.score < score_i){  currFeat.score = score_i;  }
                    for( address parentAddr : currFeat.addrs ){  frontier.push_back( parentAddr );  }
                }
            }
            i++;
        }
        return outLoss;  
    }

    double backprop( const vd& loss ){
        // Perform backpropagation similar to NN?
        // NOTE: This function assumes that `sample_parameters` and `publish_output` have already been run this iteration
        store_iter( loss ); // FIXME: DO I ACTUALLY NEED THIS?
    }
};

////////// TEST HARNESS ////////////////////////////////////////////////////////////////////////////


struct SineWave{
    // Simplest sine wave

    /// Members ///
    double ampl; // Amplitude
    double freq; // Frequency [rad/s]
    double phas; // Phase Shift
    double ofst; // Offset
    double step; // Timestep
    double t; // -- Current time

    /// Constructor ///

    SineWave( double A, double omega, double phi, double b, double ts ){
        // Set the sine wave params
        ampl = A;
        freq = omega;
        phas = phi;
        ofst = b;
        step = ts;
        t    = 0.0;
    }

    /// Methods ///

    double update( double ts = -1.0 ){
        // Advance the time and output the current function value
        t += ((ts < 0.0) ? step : ts);
        return ampl * sin( freq * t + phas ) + ofst;
    }
};


////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

int main(){

    seed_rand();

    ///// Test 1: Manually Add One Feature /////
    if( false ){
        EFB efb{ 1, 10, 5, {1.0,10.0} };
        vector<address> inAddr;
        inAddr.push_back( {0,0} );
        efb.create_feature( inAddr, ADD, 5.0, 0.5 );
        cout << "There are " << efb.layers.size() << " layers!" << endl;
        efb.load_input( {2.0} );
        efb.sample_parameters();
        efb.apply();
        efb.publish_output();
        cout << "EFB output: " << efb.get_output() << endl;
    }

    ///// Test 2: Test Harness w/ Manual Feature /////
    if( true ){
        EFB efb{ 1, 10, 5, {1.0,10.0} };
        vector<address> inAddr;
        inAddr.push_back( {0,0} );
        efb.create_feature( inAddr, ADD, 5.0, 0.5 );
        SineWave wave{ 2.0, 2*M_PI, 0.0, 0.0, 0.1 };
        // FIXME, START HERE: EXAMINE OUTPUT OF EFB
    }

    return 0;
}

/*//////// SPARE PARTS /////////////////////////////////////////////////////////////////////////////

struct EFB_Feature{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // A single operation
    // 2023-06-20: At this time, each feature has a scalar output, but the idea of batch operations makes sense
    // 2023-06-21: Members could have probably been represented as matrices at the layer level, just MVP at this point please 

    // ...

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

    // ...

};

///////////////////////////////////////////////////////////////////////////

struct EFB_Layer{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // A layer of transformed input

    // ...

    matxX get_history(){
        // Get the history of all features 
        matxX rtnArr = matxX::Zero( Nhst, 3*features.size() );
        for( uint i = 0; i < Nhst; i++ ){
            rtnArr.block( 0, i*3, Nhst, 3 ) = features[i].get_history();
        }
        return rtnArr;
    }

    // ...

};

///////////////////////////////////////////////////////////////////////////


*/