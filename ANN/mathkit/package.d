module ANN.mathkit;

import core.exception;
import std.conv; // --------------- `to!string`
import std.math.exponential; // `pow`
import std.math.algebraic; // `abs`, `sqrt` 

/// Local Imports ///
import utils;

float[] matx_mult_dyn( float[][] matx, float[] vect ){
    // Multiply the `matx` by the `vect`

    // -1. Dims check
    if( matx[0].length != vect.length )  throw new Error( 
        "Dimension mismatch!: " ~ matx[0].length.to!string ~ " -vs- " ~ vect.length.to!string
    );

    // 0. Init
    size_t  M /**/ = matx.length;
    size_t  N /**/ = matx[0].length;
    float   colSum = 0.0f;
    float[] rtnVec = alloc_array!float( M );

    // 1. Mult
    for( size_t i = 0; i < M; i++ ){
        colSum = 0.0f;
        for( size_t j = 0; j < N; j++ ){
            colSum += matx[i][j] * vect[j];
        }
        rtnVec[i] = colSum;
    }

    // N. Return
    return rtnVec;
}


float frobenius_norm( float[][] matx ){
    // 0. Init
    size_t  M /**/ = matx.length;
    size_t  N /**/ = matx[0].length;
    float   sqrSum = 0.0f;

    // 1. Sum
    for( size_t i = 0; i < M; i++ ){
        for( size_t j = 0; j < N; j++ ){
            sqrSum += pow( matx[i][j], 2 );
        }
    }

    return sqrt( sqrSum );
}