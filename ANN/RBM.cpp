/*
RBM.cpp
James Watson, 2023-04
Simplest Restricted Boltzmann Machine (RBM) example
From:
* https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
* https://www.youtube.com/watch?v=wMb7cads0go
* https://www.youtube.com/watch?v=S0kFFiHzR8M
* https://www.youtube.com/watch?v=n26NdEtma8U
* https://www.youtube.com/watch?v=iPuqoQih9xk
* Hinton, Geoffrey E. "A practical guide to training restricted Boltzmann machines." 
  Neural Networks: Tricks of the Trade: Second Edition (2012): 599-619.

Compile Command: 
g++ RBM.cpp -std=gnu++17 -I /usr/include/eigen3


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
#include <cmath>// ---- `exp` 
#include <stdlib.h> //- `srand`, `rand` 
#include <time.h> // -- `time` 
#include <algorithm> // `clamp`
using std::clamp;
#include <iostream>
using std::cout, std::endl, std::flush;
#include <vector>
using std::vector;
#include <set>
using std::set;
#include <string>
using std::string;
#include <fstream>
using std::ifstream, std::ofstream;
#include <sys/stat.h>
#include <filesystem>
#include <ctype.h> // `isspace`
#include <sstream>
using std::istringstream;

/// Eigen3 ///
#include <Eigen/Dense>
using Eigen::MatrixXd;



////////// MATH FUNCTIONS //////////////////////////////////////////////////////////////////////////

float randf(){
    // Return a pseudo-random number between 0.0 and 1.0
    return  1.0f * rand() / RAND_MAX;
}

void seed_rand(){  srand( time( NULL ) );  } // Provide a clock-based unpredictable seed to RNG

float sigmoid( float x ){  return 1.0f / (1.0f + exp(-x));  }

float Box_Muller_normal_sample( float mu = 0.0f, float sigma = 1.0f ){
    // Transform 2 uniform samples into a zero-mean normal sample
    // Source: https://www.baeldung.com/cs/uniform-to-normal-distribution
    float u1 = randf();
    float u2 = randf();
    return mu + sqrt( -2.0 * log( u1 ) ) * cos( 2.0 * M_PI * u2 ) * sigma;
}

////////// RESTRICTED BOLTZMANN MACHINE ////////////////////////////////////////////////////////////
struct RBM{ EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Simplest Restricted Boltzmann Machine 

    uint /**/ dI; // Input  dimensions
    uint /**/ dH; // Hidden dimensions
    MatrixXd W; //- Weight matrix
    MatrixXd b; //- Hidden  unit biases
    MatrixXd c; //- Visible unit biases
    MatrixXd h; //- Hidden  unit values
    MatrixXd v; //- Visible unit values
    MatrixXd x; //- Input values
    float    lr; // Learning rate

    RBM( uint inputDim, uint hiddenDim, float learnRate ){
        // Allocate arrays and init all values to zero

        // Set params
        dI = inputDim;
        dH = hiddenDim;
        lr = learnRate;
        
        // Init weights
        W = MatrixXd{ dI, dH };
        b = MatrixXd{ dH,  1 };
        h = MatrixXd{ dH,  1 };
        c = MatrixXd{ dI,  1 };
        v = MatrixXd{ dI,  1 };
        x = MatrixXd{ dI,  1 };
    }

    void random_weight_init(){
        // Set all weights and biases to normally-distributed random numbers
        float mean = 0.5;
        float var  = 0.5;
        for( uint j = 0; j < dH; j++ ){
            for( uint k = 0; k < dI; k++ ){
                W(k,j) = clamp( Box_Muller_normal_sample( mean, var ), 0.0f, 1.0f ); 
            }
            b(j,0) = clamp( Box_Muller_normal_sample( mean, var ), 0.0f, 1.0f );
        }
        for( uint k = 0; k < dI; k++ ){
            c(k,0) = clamp( Box_Muller_normal_sample( mean, var ), 0.0f, 1.0f );
        }
    }

    void set_input( const vector<float>& inputVec ){
        // Populate the input vector
        for( uint k = 0; k < dI; k++ ){  x(k,0) = inputVec[k];  }
    }

    vector<float> x_as_vec(){
        // Return `x` as a dynamic array
        vector<float> rtnX;
        for( uint k = 0; k < dI; k++ ){  rtnX.push_back( x(k,0) );  }
        return rtnX;
    }

    vector<float> v_as_vec(){
        // Return `v` as a dynamic array
        vector<float> rtnV;
        for( uint k = 0; k < dI; k++ ){  rtnV.push_back( v(k,0) );  }
        return rtnV;
    }

    void load_visible(){
        // Populate the visible units with the input vector
        for( uint k = 0; k < dI; k++ ){
            v(k,0) = x(k,0);
        }
    }

    float energy(){
        // Calc the (binary) energy function given the current network state
        float netNRG = 0.0;
        float inpNRG = 0.0;
        float hidNRG = 0.0;

        for( uint j = 0; j < dH; j++ ){
            for( uint k = 0; k < dI; k++ ){
                netNRG -= W(k,j) * h(j,0) * v(k,0);
            }
            hidNRG -= b(j,0) * h(j,0);
        }

        for( uint k = 0; k < dI; k++ ){
            inpNRG -= c(k,0) * v(k,0);
        }

        return netNRG + inpNRG + hidNRG;
    }

    float p_hj_given_v( uint j ){
        // Probability of the `j`th hidden neuron being active given the visible neuron activation
        float p_j = b(j,0);
        for( uint k = 0; k < dI; k++ ){
            p_j += v(k,0) * W(k,j);
        }
        return sigmoid( p_j );
    }

    float p_hj_given_x( uint j ){
        // Probability of the `j`th hidden neuron being active given the input vector
        float p_j = b(j,0);
        for( uint k = 0; k < dI; k++ ){
            p_j += x(k,0) * W(k,j);
        }
        return sigmoid( p_j );
    }

    float p_vk_given_h( uint k ){
        // Probability of the `j`th hidden neuron being active given the input vector
        float p_k = c(k,0);
        for( uint j = 0; j < dH; j++ ){    
            p_k += W(k,j) * h(j,0);
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
            if( randf() <= p_vk_given_h(k) )
                v(k,0) = 1.0;
            else
                v(k,0) = 0.0;
        }
    }

    void Gibbs_sample_hidden(){
        // Update the hidden neurons

        /* Section 3.1: Assuming that the hidden units are binary and that you are using Contrastive Divergence, 
        the hidden units should have stochastic binary states when they are being driven by a data-vector. 
        The probability of turning on ahiddenunit,j, is computed by applying the logistic function */
        for( uint j = 0; j < dH; j++ ){    
            if( randf() <= p_hj_given_v(j) )
                h(j,0) = 1.0;
            else
                h(j,0) = 0.0;
        }
    }

    void generate_from_input( const vector<float>& inputVec ){
        // Load the input into the visible units, then attempt to reconstruct it
        set_input( inputVec );
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
                if( x(k,0) >= 0.0 )
                    W(k,j) += lr * (p_hjx*x(k,0) - p_hjv*v(k,0));
            }
            b(j,0) += lr * (p_hjx - p_hjv);
        }

        for( uint k = 0; k < dI; k++ ){
            if( x(k,0) >= 0.0 )
                c(k,0) += lr * (x(k,0) - v(k,0));
        }
    }
};


////////// DATA PROCESSING /////////////////////////////////////////////////////////////////////////

bool file_exists( const string& fName ){ 
    // Return true if the file exists , otherwise return false
    struct stat buf; 
    if( stat( fName.c_str() , &buf ) != -1 ){ return true; } else { return false; }
}

vector<string> read_lines( string path ){ 
    // Return all the lines of text file as a string vector
    vector<string> rtnVec;
    if( file_exists( path ) ){
        ifstream fin( path ); // Open the list file for reading
        string   line; // String to store each line
        while ( std::getline( fin , line ) ){ // While we are able to fetch a line
            rtnVec.push_back( line ); // Add the file to the list to read
        }
        fin.close();
    } else { cout << "readlines: Could not open the file " << path << endl; }
    return rtnVec;
}

vector<string> split_string_ws( string input ){
    // Return a vector of strings found in `input` separated by whitespace
    vector<string> rtnWords;
    string /*---*/ currWord;
    char /*-----*/ currChar;
    size_t /*---*/ strLen = input.size();

    input.push_back( ' ' ); // Separator hack
    
    for( size_t i = 0 ; i < strLen ; i++ ){
        currChar = input[i];
        if( isspace( currChar ) ){
            if( currWord.length() > 0 )  rtnWords.push_back( currWord );
            currWord = "";
        }else{
            currWord.push_back( currChar );
        }
    }
    return rtnWords; 
}

template<typename T>
T convert_to (const string &str){
    // Convert a string to any basic type!
    // Author: Mark Holmberg, https://gist.github.com/mark-d-holmberg/862733#file-sstreamconvert-h
    istringstream ss(str);
    T /*-------*/ num;
    ss >> num;
    return num;
}

template<typename T>
vector<vector<T>> file_to_dyn_matx_ws( string fName ){
    // Read file into a 2D vector, with each row on its own line and columns separated by whitespace
    // WARNING: This function will return ragged rows if `fName` has them

    vector<T> /*---*/ row;
    vector<vector<T>> rtnMatx;
    vector<string>    tokens;
    vector<string>    lines = read_lines( fName );
    
    // For each line, separate into tokens
    for( string line : lines ){
        tokens = split_string_ws( line );
        row.clear();
        // For each token, convert and append to the row
        for( string token : tokens ){
            row.push_back( convert_to<T>( token ) );
        }
        rtnMatx.push_back( row );
    }
    return rtnMatx;
}

template<typename T>
vector<T> set_members_as_vector( const set<T>& inSet ){
    // Copy set members into a vector and return the vector
    vector<T> rtnVec;
    for( T item : inSet ){
        rtnVec.push_back( item );
    }
    return rtnVec;
}

template<typename T>
long get_index_of_key_in_vec( const vector<T>& vec, const T& key ){
    // Return the index of the search key, or -1 if DNE
    // Author: Erick Lumunge, https://iq.opengenus.org/find-element-in-vector-cpp-stl/
    typename vector<T>::const_iterator it;
    it = find( vec.begin(), vec.end(), key );
    if( it != vec.end() )
        return (it - vec.begin());
    else
        return -1;
}

vector<vector<float>> movie_data_to_user_vectors( string fName, string headingFname = "" ){
    // This matrix will have the *users as the rows* and *the movies as the columns*.
    
    /// Vector Heading Init ///
    bool     headingsProvided = (headingFname.length() > 0);
    ifstream headingsIn;
    ofstream headingsOut;
    if( !headingsProvided ){
        headingsOut.open( "columnHeadings.txt" ); 
    }else{
        headingsIn = ifstream{ headingFname };
    }

    // Read movie data as plaintext integers
    vector<vector<long>> movieData = file_to_dyn_matx_ws<long>( fName );
    set<long> /*------*/ userID;
    set<long> /*------*/ moviID;
    vector<long> /*---*/ userList;
    vector<long> /*---*/ moviList;
    vector<vector<long>> movieHeadings;

    if( headingsProvided ){
        for( vector<long> row : movieData ){  userID.insert( row[0] );  }
        movieHeadings = file_to_dyn_matx_ws<long>( headingFname );
        moviList /**/ = movieHeadings[0];
    }else{
        if( headingsOut ){
            for( vector<long> row : movieData ){
                userID.insert( row[0] );
                moviID.insert( row[1] );
            }
            moviList = set_members_as_vector<long>( moviID );
            for( long mID : moviList ){  headingsOut << mID << " ";  }
            headingsOut.close();
        }else  cout << "Could not open the default headings file!" << endl;
    }
    userList = set_members_as_vector<long>( userID );

    ulong N_users = userList.size();
    ulong N_movis = moviList.size();
    cout << "There are " << N_users << " unique users."  << endl;
    cout << "There are " << N_movis << " unique movies." << endl;

    /// Create Training Vectors ///
    vector<vector<float>> trainData;
    vector<float> /*---*/ ratingRow;
    long /*------------*/ founDex  = 0;
    long /*------------*/ currMovi = 0;
    long /*------------*/ rating   = 0;
    for( long user : userList ){
        ratingRow.clear();
        for( ulong i = 0; i < N_movis; i++ ){  ratingRow.push_back( -1.0f );  }
        for( vector<long> row : movieData ){
            if( row[0] == user ){
                currMovi = row[1];
                founDex  = get_index_of_key_in_vec<long>( moviList, currMovi );
                rating   = row[2];
                if( founDex > -1 )  ratingRow[ founDex ] = (rating < 3 ? 0.0f : 1.0f);
            }
        }
        trainData.push_back( ratingRow );
    }
    cout << "Created training data with " << trainData.size() << " rows and " << 
             trainData[0].size() << " columns!" << endl;
    return trainData;
}

float fraction_same_nonnegative( const vector<float>& An, const vector<float>& B ){
    // `An` is a vector that might have negative values, calc the similarity between the two where `An` is non-negative
    size_t N = An.size();
    size_t L = 0;
    size_t S = 0;
    if( N != B.size() ){  return -1.0f;  }else{
        for( size_t i = 0; i < N; i++ ){
            if( An[i] >= 0.0f ){
                L++;
                if( An[i] == B[i] )  S++;
            }
        }
        return ((float) S) / ((float) L);
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

int main(){
    seed_rand();

    // RBM net = RBM( 5, 5, 0.5 );
    // net.random_weight_init();
    // cout << net.W << endl << endl; // 2023-04-04: Appropriately random!
    // cout << net.b << endl << endl;
    // cout << net.c << endl << endl;

    vector<vector<float>> trainData = movie_data_to_user_vectors( "../Data/ml-100k/u1.base" );
    vector<vector<float>> testData  = movie_data_to_user_vectors( "../Data/ml-100k/u1.test", "columnHeadings.txt" );

    /// Create RBM ///
    RBM  net     = RBM( trainData[0].size(), 100, 0.025 );
    uint N_epoch = 10;
    uint div     = trainData.size() / 100;
    net.random_weight_init();

    /// Train ///
    cout << "##### Training #####" << endl;
    for( uint i = 0; i < N_epoch; i++ ){
        cout << "## Epoch " << i+1 << " ##" << endl;
        for( ulong j = 0; j < trainData.size(); j++ ){
            net.set_input( trainData[j] );
            net.Contrastive_Divergence_iter();
            if( j%div == 0 ){  cout << "." << flush;  }
        }
        cout << endl << "Network Energy: " << net.energy() << endl;
    }

    /// Test ///
    float fracCorrect = 0.0f;
    div = testData.size() / 100;

    cout << "##### Validation #####" << endl;
    for( ulong j = 0; j < testData.size(); j++ ){
        net.generate_from_input( testData[j] );
        fracCorrect += fraction_same_nonnegative( net.x_as_vec(), net.v_as_vec() ); // Only count accuracy for rated movies
        if( j%div == 0 ){  cout << "." << flush;  }
    }

    fracCorrect /= (float) testData.size();
    cout << endl << fracCorrect << endl; // 0.819597, YAY

    return 0;
}