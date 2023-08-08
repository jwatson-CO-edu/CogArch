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
#include <algorithm>
using std::sort;
#include <deque>
using std::deque;
#include <iomanip>

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

double percent_difference( double op1, double op2 ){
    // Calc the Percent Difference according to https://www.mathsisfun.com/percentage-difference.html
    return abs( op1 - op2 ) / abs( (op1+op2)/2.0 ) * 100.0;
}

void set_cout_precision( uint decPlaces ){
    // Set the decimal precision of displayed floating point numbers when printing to the terminal
    std::cout << std::fixed;
    std::cout << std::setprecision( decPlaces );
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
uint EFBF_No = 0;

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
    double /*----*/ lr; // --- Learning rate, NOT USED?
    double /*----*/ output; // Output 
    uint /*------*/ Nhst; // - Steps of history to retain
    deque<double>   vHst; // - History of values, used for Delay
    double /*----*/ score; //- Fitness value of this feature
    double /*----*/ accum; //- Accumulated fitness for this episode
    uint /*------*/ Nruns; //- Number of calculations for this episode
    uint /*------*/ ID;
    
    /// Constructor ///

    EFB_Feature(){
        // Default constructor
        opType = NOP;
        param  =  0.0;
        Nhst   = 10;
        lr     =  0.001;
        score  = -1e9;
        accum  = 0.0;
        Nruns  = 0;
        ID     = EFBF_No;
        ++EFBF_No;
    }

    /// Methods ///

    double set_score( double s ){
        // Set the current score and return the current average score
        score = s;
        accum += score;
        Nruns++;
        // cout << "Score set for feature " << ID << ": " << score << endl;
        return accum / (1.0*Nruns);
    }

    double get_avg_score(){  return accum / (1.0*Nruns);  } // Return the current average score

    void reset_score(){  score = -1e9;  } // Reset the score to some low value

    double reset_fitness(){
        // Wipe score and average and return the previous average score
        double rtnAvg = accum / (1.0*Nruns);
        score = -1e9;
        accum = 0.0;
        Nruns = 0;
        return rtnAvg;
    }

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
        bool _DEBUG = false;
        output = 0.0;
        uint delay;
        switch( opType ){
            case ADD: // Add
                if( _DEBUG )  cout << "Apply ADD ..." << endl;
                for( double operand : operands ){  output += operand;  }
                output += param;
                break;
            case SUB: // Subtract
                if( _DEBUG )  cout << "Apply SUB ..." << endl;
                output = operands[0];
                for( size_t i = 1; i < operands.size(); i++ ){  output -= operands[i];  }
                output -= param;
                break;
            case CON: // Constant
                if( _DEBUG )  cout << "Apply CON ..." << endl;
                output = param;
                break;
            case DLY: //Delay
                if( _DEBUG )  cout << "Apply DLY ..." << endl;
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
                if( _DEBUG )  cout << "Apply NOP ..." << endl;
            default:
                if( _DEBUG )  cout << "Apply `default` ..." << endl;
                // output = 0.0;
        }
        if( _DEBUG )  cout << "Feature Output: " << output << endl;
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
        // if( x.size() != dI )  cout << "Input dimension " << x.size() << ", but expected " << dI << " !" << endl;
    }

    uint size_features(){
        // Return the number of total features
        return addrs.size() - dI;
    }

    uint size_addrs(){
        // Return the number of valid addresses
        return addrs.size(); // The network inputs are valid addresses
    }

    bool create_feature( const vector<address>& opAddrs, EFB_Op typ, double paramAvg, double pVariance ){
        // Create a new feature, assuming the decision to create the feature has already been made
        // 0. Init
        EFB_Feature rtnFeature;
        bool /*--*/ placed = false;
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
        // 6. Do not violate the 
        if( nxtLvl > dpthMax ){
            return false;
        }else{
            if(nxtLvl > layers.size())  layers.push_back( EFB_Layer{} );
            layers[ nxtLvl-1 ].features.push_back( rtnFeature );
            layers[ nxtLvl-1 ].y.push_back( 0.0 );
            addrs.push_back( { nxtLvl, (uint) (layers[ nxtLvl-1 ].features.size()-1) } );
            return true;
        }
    }

    vector<address> rand_unique_addresses( uint Noper = 2, uint maxAttempts = 50 ){
        // Return `Noper` non-repeating addresses that exist on the bus
        vector<address> opAddrs;
        uint /*------*/ Naddr   = addrs.size();
        uint /*------*/ mxTry   = maxAttempts;
        address /*---*/ oneAddr = addrs[ randu(0, Naddr-1) ];
        uint /*------*/ j;
        for( uint i = 0; i < Noper; i++ ){
            oneAddr = addrs[ randu(0, Naddr-1) ];
            j /*-*/ = 0;
            while( (is_arg_in_vec( oneAddr, opAddrs ) || (oneAddr[0] >= dpthMax)) && ( j < mxTry ) ){  
                oneAddr = addrs[ randu(0, Naddr-1) ];
                j++;  
            }
            if( j >= mxTry )  return opAddrs;
            opAddrs.push_back( oneAddr );
        }
        return opAddrs;
    }

    bool create_feature(){
        // Create a new feature, assuming the decision to create the feature has already been made
        // 0. Init
        EFB_Feature rtnFeature;
        bool /*--*/ placed = false;
        // 1. Choose the number of operands && Choose the addresses
        vector<address> opAddrs = rand_unique_addresses( randu( 1, min( operMax, (uint) addrs.size() ) ) );
        // 3. Attempt feature creation
        return create_feature( 
            opAddrs, 
            static_cast<EFB_Op>( randu(1, 4) ), 
            randd( paramRange[0], paramRange[1] ), 
            paramVar
        );
    }

    vd fetch_input( const vector<address>& addrs ){
        // Get the necessary inputs to this feature
        // NOTE: Layer 0 is the EFB input
        vd   rtnArr;
        // cout << "Retrieve Operands: " << addrs << endl;
        for( array<uint,2> addr_i : addrs ){
            if( addr_i[0] == 0 ){
                rtnArr.push_back( x[ addr_i[1] ] );
            }else{
                rtnArr.push_back( layers[ addr_i[0]-1 ].y[ addr_i[1] ] );
            }
        }
        // cout << "Input Obtained!: " << rtnArr << endl;
        return rtnArr;
    }

    void sample_parameters(){
        // Set the parameter
        for( EFB_Layer& layer : layers ){
            for( EFB_Feature& feature : layer.features ){  feature.sample_parameter();  }
        }
    }

    void apply(){
        // Run all features
        // 1. For each layer
        for( EFB_Layer& layer : layers ){
            // 2. Fire all features in the layer
            for( EFB_Feature& feature : layer.features ){
                feature.output = feature.apply( fetch_input( feature.addrs ) );
                // cout << "Feature stored: " << feature.output << endl;
            }
            // 3. The output needs to be advertized at the layer, so that it can be fetched by successive layers
            uint i = 0;
            for( EFB_Feature& feature : layer.features ){
                layer.y[i] = feature.output;
                i++;
            }
            // cout << "Published: " << layer.y << endl;
        }
    }

    vd get_output(){
        // Return the output of the last layer
        return layers.back().y;
    }

    vd forward( const vd& input ){
        // Run one input --to-> output cycle and return
        load_input( input );
        // sample_parameters();
        apply();
        return get_output();
    }

    vd get_output_params(){
        // Get the parameters of the output layer for this iteration
        vd rtnParams;
        for( EFB_Feature& feature : layers.back().features ){  rtnParams.push_back( feature.output );  }  
        return rtnParams;
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
            score_i = 100.0 - percent_difference( signalTs, Yfeat );
            // 3. Update parents
            // frontier.push_back( {(uint)layers.size()-1, (uint)i} );
            frontier.push_back( {(uint)layers.size(), (uint)i} );
            while( frontier.size() ){
                addr_j = frontier.front();
                frontier.pop_front();
                if( addr_j[0] > 0 ){
                    EFB_Feature& currFeat = layers[ addr_j[0]-1 ].features[ addr_j[1] ];
                    if( currFeat.score < score_i ){  currFeat.set_score( score_i );  }
                    // currFeat.set_score( score_i );
                    for( address parentAddr : currFeat.addrs ){  frontier.push_back( parentAddr );  }
                }
            }
            i++;
        }
        return outLoss;  
    }

    vd get_avg_score(){
        vd fitness;
        for( EFB_Feature& feature : layers.back().features ){
            fitness.push_back( feature.get_avg_score() );
        }
        return fitness;
    }

    EFB_Feature* get_at_addr( address ad ){
        // Get a pointer to the feature at the particular address
        if( ad[0] > 0 ){
            return &(layers[ad[0]-1].features[ad[1]]);
        }else{
            return nullptr;
        }
    }

    void reset_score(){
        // Set all scores to some low number
        for( EFB_Layer& layer : layers ){
            for( EFB_Feature& feature : layer.features ){  feature.reset_score();  }
        }
    }

    void cull_and_replace_pop( double cullFrac = 0.25 ){
        // Destroy members that have the lowest score and replace them with new elements

        // 0. Calc how many features to replace
        uint Ncul = (uint) size_features() * cullFrac;

        // 1. Create a vector of pairs of (Feature Address, Feature Score)
        vector<pair<address,double>> ranking;
        for( address ad : addrs ){
            if( ad[0] > 0 ){
                ranking.push_back( { ad , get_at_addr( ad )->reset_fitness() } );
            }
        }

        // 2. Sort created features by ascending score
        sort(
            ranking.begin(), 
            ranking.end(), 
            [](const std::pair<address,double> &left, const std::pair<address,double> &right){
                return left.second < right.second;
            }
        );

        // PRINT TO TS CORRECT ORDER
        // cout << endl;
        // for( pair<address,double> p : ranking ){
        //     cout << p.first << ": " << p.second << endl;
        // }
        // cout << endl;

        // 3. Kill the first `Ncul` features and replace
        // FIXME, START HERE: CULL AND REPLACE
        
        for( uint i = 0; i < Ncul; ++i ){
            // A. Fetch feature
            EFB_Feature* currOp = get_at_addr( ranking[i].first );
            // B. In most cases replace the feature entirely
            if( randd() >= cullFrac ){
                // i. Set new operands
                currOp->addrs.clear();
                currOp->addrs = rand_unique_addresses( randu( 1, min( operMax, (uint) addrs.size() ) ) );
                // ii. Set new type
                currOp->opType = static_cast<EFB_Op>( randu(1, 4) );    
            }
            // C. Else, adjust the param only, 2023-08-08: Not descending params at this time
            // iii. Set new parameter in every case
            currOp->sample_parameter();
        }
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
    set_cout_precision( 4 );

    ///// Test 1: Manually Add One Feature /////
    if( false ){
        EFB /*-------*/ efb{ 1, 10, 5, {1.0,10.0} };
        vector<address> inAddr;
        inAddr.push_back( {0,0} );
        efb.create_feature( inAddr, ADD, 5.0, 0.5 );
        // cout << "There are " << efb.layers.size() << " layers!" << endl;
        efb.load_input( {2.0} );
        efb.sample_parameters();
        efb.apply();
        // cout << "EFB output: " << efb.get_output() << endl;
    }

    ///// Test 2: Test Harness w/ Manual Feature /////
    if( false ){
        EFB /*-------*/ efb{ 1, 10, 5, {1.0,10.0} };
        vector<address> inAddr;
        uint /*------*/ N = 100;
        double /*----*/ val;
        double /*----*/ out;
        SineWave /*--*/ wave{ 2.0, 2*M_PI, 0.0, 0.0, 0.05 };
        inAddr.push_back( {0,0} );
        efb.create_feature( inAddr, ADD, 5.0, 0.5 );
        for( uint i = 0; i < N; i++ ){  
            val = wave.update();
            efb.load_input( {val} );
            efb.sample_parameters();
            efb.apply();
            out = efb.get_output()[0];
            cout << val << "\t--efb->\t" << out << endl;
        }  
    }

    ///// Test 3: Create a Random Feature /////
    if( false ){
        /// Init ///
        EFB /**/ efb{ 1, 10, 5, {1.0,10.0} };
        uint     N = 100;
        double   val;
        double   out;
        SineWave wave{ 2.0, 2*M_PI, 0.0, 0.0, 0.05 };
        /// Create Feature ///
        efb.create_feature();
        efb.sample_parameters();
        for( uint i = 0; i < N; i++ ){  
            val = wave.update();
            efb.load_input( {val} );
            efb.apply();
            out = efb.get_output()[0];
            cout << val << "\t--efb->\t" << out << endl;
        }  
    }

    ///// Test 4: Calc Fitness of a Random Feature /////
    if( false ){
        /// Init ///
        EFB /**/ efb{ 1, 10, 5, {1.0,10.0} };
        uint     N = 100;
        double   val, out, tru;
        vd /*-*/ loss;
        SineWave inpt{ 2.0, 2*M_PI, 0.0 , 0.0, 0.05 };
        SineWave outp{ 2.0, 2*M_PI, M_PI, 0.0, 0.05 };
        /// Create Feature ///
        efb.create_feature();
        efb.sample_parameters();
        for( uint i = 0; i < N; i++ ){  
            val = inpt.update();
            tru = outp.update();
            efb.load_input( {val} );
            efb.apply();
            loss = efb.score_output_and_calc_loss( tru );
            out = efb.get_output()[0];
            cout << val << "\t--efb->\t" << out << " -vs- " << tru << endl;
            cout << "Loss: " << loss << endl;
        }  
        cout << "Avg. Fitness: " << efb.get_avg_score() << endl;
    }

    ///// Test 5: Multiple Random Feature /////
    if( false ){
        /// Init ///
        uint     N = 100; // Number of iterations
        uint     M = 100; // Number of features
        EFB /**/ efb{ 1, M, 15, {1.0,10.0} };
        double   val, tru;
        vd /*-*/ loss, out;
        SineWave inpt{ 2.0, 2*M_PI, 0.0 , 0.0, 0.05 };
        SineWave outp{ 2.0, 2*M_PI, M_PI, 0.0, 0.05 };
        /// Create Feature(s) ///
        for( uint i = 0; i < M; ++i ){  efb.create_feature();  }
        efb.sample_parameters();
        for( uint i = 0; i < N; i++ ){  
            val = inpt.update();
            tru = outp.update();
            efb.load_input( {val} );
            efb.apply();
            loss = efb.score_output_and_calc_loss( tru );
            out  = efb.get_output();
            cout << val << "\t--efb->\t" << out << " -vs- " << tru << endl;
            cout << "Loss: " << loss << endl;
        }  
        cout << endl << endl << "Avg. Fitness: " << efb.get_avg_score() << endl;
    }

    // FIXME: ENFORCE THE CORRECT NUMBER OF FEATURES, RETRY ON FAILED CREATION

    ///// Test 6: Culling and Replication /////
    if( true ){
        /// Init ///
        uint     N = 100; // Number of iterations
        uint     M = 100; // Number of features
        uint     Q =  25; // Number of generations
        EFB /**/ efb{ 1, M, 15, {1.0,10.0} };
        double   val, tru;
        vd /*-*/ loss, out;
        SineWave inpt{ 2.0, 2*M_PI, 0.0 , 0.0, 0.05 };
        SineWave outp{ 2.0, 2*M_PI, M_PI, 0.0, 0.05 };
        /// Create Feature(s) ///
        for( uint i = 0; i < M; ++i ){  efb.create_feature();  }
        efb.sample_parameters();

        for( uint j = 0; j < Q; ++j ){
            for( uint i = 0; i < N; i++ ){  
                val = inpt.update();
                tru = outp.update();
                efb.load_input( {val} );
                efb.apply();
                efb.score_output_and_calc_loss( tru );
                efb.get_output();
            }  
            cout << endl << endl << "Avg. Fitness: " << efb.get_avg_score() << endl;
            efb.cull_and_replace_pop( 0.25 );
            efb.reset_score();
        }
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

struct EFB{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // ...

    void store_iter( const vd& loss ){
        // Store info for this iteration
        // NOTE: This function assumes that `sample_parameters` and `publish_output` have already been run this iteration
        const EFB_Layer& lastLayer = layers.back();
        vHst.push_front( get_output() );
        lHst.push_front( loss );
        pHst.push_front( get_output_params() );
    }

    double backprop( const vd& loss ){
        // Perform backpropagation similar to NN?
        // NOTE: This function assumes that `sample_parameters` and `publish_output` have already been run this iteration
        store_iter( loss ); // FIXME: DO I ACTUALLY NEED THIS?
    }

    // ...

}

///////////////////////////////////////////////////////////////////////////


*/