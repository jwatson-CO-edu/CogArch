module ANN.mathkit;

import core.exception;
import std.conv; // --------------- `to!string`

/// Local Imports ///
import ANN.utils;

float[] matx_mult_dyn( float[][] matx, float[] vect ){
    // Multiply the `matx` by the `vect`

    // -1. Dims check
    if( matx[0].length != vect.length )  throw new ArrayIndexError( 
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