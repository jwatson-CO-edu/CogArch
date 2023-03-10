module RBM;

/*
RBM.d
James Watson, 2023-02
Simplest Restricted Boltzmann Machine (RBM) example
From:
* https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
* https://www.youtube.com/watch?v=wMb7cads0go
* https://www.youtube.com/watch?v=S0kFFiHzR8M
* https://www.youtube.com/watch?v=n26NdEtma8U
* https://www.youtube.com/watch?v=iPuqoQih9xk
* Hinton, Geoffrey E. "A practical guide to training restricted Boltzmann machines." 
  Neural Networks: Tricks of the Trade: Second Edition (2012): 599-619.

Compile & Run:
rdmd RBM.d


///// Background /////

Boltzmann Machine is a generative unsupervised model, which involves learning a probability distribution 
from an original dataset and using it to make inferences about never before seen data.

Boltzmann Machine has an input layer (also referred to as the visible layer) and one or several hidden layers.
Neurons generate information regardless they are hidden or visible.

RBM’s objective is to find the joint probability distribution that maximizes the log-likelihood function.

Since RBMs are undirected, they don’t adjust their weights through gradient descent and backpropagation. 
They adjust their weights through a process called contrastive divergence. At the start of this process, 
weights for the visible nodes are randomly generated and used to generate the hidden nodes. 
These hidden nodes then use the same weights to reconstruct visible nodes. The weights used to 
reconstruct the visible nodes are the same throughout.

Now, the difference v(0)−v(1) can be considered as the reconstruction error that we need to reduce 
in subsequent steps of the training process. So the weights are adjusted in each iteration so as to 
minimize this error.

*/

////////// INIT ////////////////////////////////////////////////////////////////////////////////////

/// Standard Imports ///
import std.stdio; // -------------- `writeln`
import core.stdc.stdlib; // ------- `malloc`
import std.math.exponential; // --- `exp`
import std.random; // ------------- RNG
import std.conv; // --------------- `to!string`
import std.algorithm.searching; //- `countUntil`, linear search
import std.algorithm.comparison; // `clamp`
import std.math.exponential; // --- Natural `log`
import std.math; // --------------- `cos`, `sqrt`


/// Local Imports ///
import utils;

/// Randomness ///
Mt19937 rnd;



////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////


float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }


float Box_Muller_normal_sample( float mu = 0.0f, float sigma = 1.0f ){
    // Transform 2 uniform samples into a zero-mean normal sample
    // Source: https://www.baeldung.com/cs/uniform-to-normal-distribution
    float u1 = uniform( 0.0f, 1.0f, rnd );
    float u2 = uniform( 0.0f, 1.0f, rnd );
    return mu + sqrt( -2.0 * log( u1 ) ) * cos( 2.0 * PI * u2 ) * sigma;
}



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
        W = alloc_2D_dyn_array!float( dI, dH );
        // Init hidden units
        b = alloc_dyn_array!float( dH );
        h = alloc_dyn_array!float( dH );
        // Init input units
        c = alloc_dyn_array!float( dI );
        v = alloc_dyn_array!float( dI );
        x = alloc_dyn_array!float( dI );
    }


    void random_weight_init(){
        // Set all weights and biases to normally-distributed random numbers
        

        float mean = 0.5;
        float var  = 0.5;

        for( uint j = 0; j < dH; j++ ){
            for( uint k = 0; k < dI; k++ ){
                W[k][j] = clamp( Box_Muller_normal_sample( mean, var ), 0.0f, 1.0f ); 
            }
            b[j] = clamp( Box_Muller_normal_sample( mean, var ), 0.0f, 1.0f );
        }

        for( uint k = 0; k < dI; k++ ){
            c[k] = clamp( Box_Muller_normal_sample( mean, var ), 0.0f, 1.0f );
        }
    }


    void set_input( float[] input ){
        // Populate the input vector
        for( uint k = 0; k < dI; k++ ){
            x[k] = input[k];
            // writeln( x );
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

        writeln( "\t" ~ netNRG.to!string );
        writeln( "\t" ~ inpNRG.to!string );
        writeln( "\t" ~ hidNRG.to!string );

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


    void generate_from_input( float[] input ){
        // Load the input into the visible units, then attempt to reconstruct it
        set_input( input );
        load_visible();
        Gibbs_sample_hidden();
        Gibbs_sample_visible();
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
                if( x[k] >= 0.0 )
                    W[k][j] += lr * (p_hjx*x[k] - p_hjv*v[k]);
            }
            b[j] += lr * (p_hjx - p_hjv);
        }

        for( uint k = 0; k < dI; k++ ){
            if( x[k] >= 0.0 )
                c[k] += lr * (x[k] - v[k]);
        }
    }
}


////////// MAIN ////////////////////////////////////////////////////////////////////////////////////

/* ///// Problem /////

Let us assume that some people were asked to rate a set of movies on a scale of 1–5 and each movie 
could be explained in terms of a set of latent factors such as drama, fantasy, action and many more.
Given the inputs, the RMB then tries to discover latent factors in the data that can explain the 
movie choices and each hidden neuron represents one of the latent factors.

After the training phase, the goal is to predict a binary rating for the movies that had not been seen yet. 
Given the *training data of a specific user*, 
the network is able to identify the latent factors based on the user’s preference and sample from 
Bernoulli distribution can be used to find out which of the visible neurons now become active.

1. Train the network on the data of all users
2. During inference-time, take the training data of a specific user
3. Use this data to obtain the activations of hidden neurons
4. Use the hidden neuron values to get the activations of input neurons
5. The new values of input neurons show the rating the user would give yet unseen movies

In order to build the RBM, we need a matrix with the users’ ratings. 
This matrix will have the 
*users as the rows* 
and 
*the movies as the columns*.

We use -1 to represent movies that a user never rated. We then convert the ratings that were 
*rated 1 and 2 to 0*
and movies that were rated 
*3, 4 and, 5 to 1*. 
We do this for both the test set and the training set.


///// Architecture /////
We initialize the weight and bias. 
We do this randomly *using a normal distribution*.
Now we set the 
*number of visible nodes to the length of the input vector* 
and 
*the number of hidden nodes to 100*.

*There are 10 epochs*
*/


float[][] movie_data_to_user_vectors( string fName, string headingFname = "" ){
    // This matrix will have the *users as the rows* and *the movies as the columns*.

    /// Vector Heading Init ///
    bool headingsProvided = (headingFname.length > 0);
    File headingFile;
    if( !headingsProvided )  headingFile = File( "columnHeadings.txt", "w" );

    /// Load Data ///
    long[][] movieData = file_to_dyn_matx_ws!long( fName );
    Set!long userID    = Set!long();
    Set!long moviID    = Set!long();
    long[]   userList;
    long[]   moviList;
    long[][] movieHeadings;

    if( headingsProvided ){
        foreach( long[] row; movieData ){
            userID.add( row[0] );
        } 
        movieHeadings = file_to_dyn_matx_ws!long( headingFname );
        moviList      = movieHeadings[0];
    }else{
        foreach( long[] row; movieData ){
            userID.add( row[0] );
            moviID.add( row[1] );
        } 
        moviList = moviID.get_members();
        foreach( long mID; moviList ){
            headingFile.write( mID.to!string ~ " ");
        }
        headingFile.close();
    }
    userList = userID.get_members();

    ulong  N_users = userList.length;
    ulong  N_movis = moviList.length;
    writeln( "There are " ~ N_users.to!string ~ " unique users." );
    writeln( "There are " ~ N_movis.to!string ~ " unique movies." );
    /* There are  943 unique users.
       There are 1650 unique movies. */

    /// Create Training Vectors ///
    float[][] trainData;
    float[]   ratingRow;
    long     founDex  = 0;
    long     currMovi = 0;
    long     rating   = 0;
    foreach( long user; userList ){
        ratingRow = [];
        for( ulong i = 0; i < N_movis; i++ )  ratingRow ~= -1.0f;
        foreach( long[] row; movieData ){
            if( row[0] == user ){
                currMovi = row[1];
                founDex  = moviList.countUntil!( m => m == currMovi );
                rating   = row[2];
                if( founDex > -1 )  ratingRow[ founDex ] = (rating < 3 ? 0.0f : 1.0f);
            }
        }
        trainData ~= ratingRow;
    }
    writeln( "Created training data with " ~ trainData.length.to!string ~ " rows and " ~ 
             trainData[0].length.to!string ~ " columns!"  );

    return trainData;
}

float fraction_same_nonnegative( float[] An, float[] B ){
    // `An` is a vector that might have negative values, calc the similarity between the two where `An` is non-negative
    size_t N = An.length;
    size_t L = 0;
    size_t S = 0;
    if( An.length != B.length ){  return -1.0f;  }
    else{
        for( size_t i = 0; i < N; i++ ){
            if( An[i] >= 0.0f ){
                L++;
                if( An[i] == B[i] )  S++;
            }
        }
        return (cast(float) S) / (cast(float) L);
    }
}


void main(){
    /// Init Random ///
    rnd = Random( unpredictableSeed );

    float[][] trainData = movie_data_to_user_vectors( "../Data/ml-100k/u1.base" );
    // writeln( trainData[0] );
    float[][] testData  = movie_data_to_user_vectors( "../Data/ml-100k/u1.test", "columnHeadings.txt" );

    /// Create RBM ///
    RBM  net     = RBM( cast(uint) trainData[0].length, 100, 0.025 );
    uint N_epoch = 10;

    net.random_weight_init();
    writeln( net.W[0] );

    /// Train ///
    for( uint i = 0; i < N_epoch; i++ ){
        for( ulong j = 0; j < trainData.length; j++ ){
            net.set_input( trainData[j] );
            // net.load_visible();
            net.Contrastive_Divergence_iter();
        }
        writeln( net.energy() );
    }

    /// Test ///
    float fracCorrect = 0.0f;

    for( ulong j = 0; j < testData.length; j++ ){
        net.generate_from_input( testData[j] );
        fracCorrect += fraction_same_nonnegative( net.x, net.v ); // Only count accuracy for rated movies
        writeln( fracCorrect );
    }

    fracCorrect /= cast(float) testData.length;
    writeln();
    writeln( fracCorrect ); // 0.81174, YAY
}