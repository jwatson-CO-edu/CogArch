module RBM;

/*
RBM.d
James Watson, 2023-02
Simplest Restricted Boltzmann Machine (RBM) example
From:
* https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
* Hinton, Geoffrey E. "A practical guide to training restricted Boltzmann machines." 
  Neural Networks: Tricks of the Trade: Second Edition (2012): 599-619.


rdmd RBM.d
*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

import std.stdio; // ---------- `writeln`
import core.stdc.stdlib; // --- `malloc`
import std.math.exponential; // `exp`
import std.random;

// Randomness
Mt19937 rnd;




////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////

float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }



////////// RESTRICTED BOLTZMANN MACHINE ////////////////////////////////////////////////////////////

struct RBM{
    // Simplest Restricted Boltzmann Machine 

    uint /**/ dI; // Input  dimensions
    uint /**/ dH; // Hidden dimensions
    float[][] W; //- Weight matrix
    float[]   b; //- Hidden  unit biases
    float[]   c; //- Visible unit biases
    float[]   h; //- Hidden  unit values
    float[]   v; //- Visible unit values
    float[]   x; //- Input values
    float     lr; // Learning rate

    this( uint inputDim, uint hiddenDim, float learnRate ){
        // Allocate arrays and init all values to zero

        // Set params
        dI = inputDim;
        dH = hiddenDim;
        lr = learnRate;

        // Init weights
        W = alloc_2D_array!float( dI, dH );
        // Init hidden units
        b = alloc_array!float( dH );
        h = alloc_array!float( dH );
        // Init input units
        c = alloc_array!float( dI );
        v = alloc_array!float( dI );
        x = alloc_array!float( dI );
    }


    void set_input( float[] input ){
        // Populate the input vector
        for( uint k = 0; k < dI; k++ ){
            x[k] = input[k];
        }
    }


    void load_visible(){
        // Populate the visible units with the input vector
        for( uint k = 0; k < dI; k++ ){
            v[k] = x[k];
        }
    }


    float energy(){
        // Calc the (binary) energy function given the current network state
        float netNRG = 0.0;
        float inpNRG = 0.0;
        float hidNRG = 0.0;

        for( uint j = 0; j < dH; j++ ){
            for( uint k = 0; k < dI; k++ ){
                netNRG -= W[k][j] * h[j] * v[k];
            }
            hidNRG -= b[j] * h[j];
        }

        for( uint k = 0; k < dI; k++ ){
            inpNRG -= c[k] * v[k];
        }

        return netNRG + inpNRG + hidNRG;
    }


    float p_hj_given_v( uint j ){
        // Probability of the `j`th hidden neuron being active given the visible neuron activation
        float p_j = b[j];
        for( uint k = 0; k < dI; k++ ){
            p_j += v[k] * W[k][j];
        }
        return sigmoid( p_j );
    }


    float p_hj_given_x( uint j ){
        // Probability of the `j`th hidden neuron being active given the input vector
        float p_j = b[j];
        for( uint k = 0; k < dI; k++ ){
            p_j += x[k] * W[k][j];
        }
        return sigmoid( p_j );
    }


    float p_vk_given_h( uint k ){
        // Probability of the `j`th hidden neuron being active given the input vector
        float p_k = c[k];
        for( uint j = 0; j < dH; j++ ){    
            p_k += W[k][j] * h[j];
        }
        return sigmoid( p_k );
    }


    float p_h_given_v(){
        // Conditional probabolity of the current hidden values given the current input
        float rtnProd = 1.0;
        for( uint j = 0; j < dH; j++ ){
            rtnProd *= p_hj_given_v(j);
        }
        return rtnProd;
    }


    float p_v_given_h(){
        // Conditional probabolity of the current input given the current hidden values
        float rtnProd = 1.0;
        for( uint k = 0; k < dI; k++ ){
            rtnProd *= p_vk_given_h(k);
        }
        return rtnProd;
    }


    void Gibbs_sample_visible(){
        // Run Gibbs sampling on all visibile units
        for( uint k = 0; k < dI; k++ ){
            if( uniform( 0.0f, 1.0f, rnd ) <= p_vk_given_h(k) )
                v[k] = 1.0;
            else
                v[k] = 0.0;
        }
    }


    void Gibbs_sample_hidden(){
        // Update the hidden neurons

        /* Section 3.1: Assuming that the hidden units are binary and that you are using Contrastive Divergence, 
        the hidden units should have stochastic binary states when they are being driven by a data-vector. 
        The probability of turning on ahiddenunit,j, is computed by applying the logistic function */
        for( uint j = 0; j < dH; j++ ){    
            if( uniform( 0.0f, 1.0f, rnd ) <= p_hj_given_v(j) )
                h[j] = 1.0;
            else
                h[j] = 0.0;
        }
    }


    


    void Contrastive_Divergence_iter(){
        // Run a single iteration of (Persistent) Contrastive Divergence
        float p_hjv = 0.0;
        float p_hjx = 0.0;


        // 1. Generate a negative sample using n steps of Gibbs sampling
        /* One iteration of alternating Gibbs sampling consists of updating all of the hidden units 
        in parallel followed by updating all of the visible units in parallel. */
        Gibbs_sample_hidden();
        Gibbs_sample_visible();

        // 2. Update parameters
        for( uint j = 0; j < dH; j++ ){
            p_hjv = p_hj_given_v(j);
            p_hjx = p_hj_given_x(j);
            for( uint k = 0; k < dI; k++ ){
                W[k][j] += lr * (p_hjx*x[k] - p_hjv*v[k]);
            }
            b[j] += lr * (p_hjx - p_hjv);
        }

        for( uint k = 0; k < dI; k++ ){
            c[k] += lr * (x[k] - v[k]);
        }
    }
}


////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

void main(){
    rnd = Random( unpredictableSeed );

}