module aa_loadTest;

import std.stdio;

import utils;

void main(){
    uint[][] movieData = file_to_matx_ws!uint( "../Data/ml-100k/u1.base" );
    // FIXME, START HERE: Found file, but FAILED TO LOAD DATA?
    writeln( movieData.length );
    writeln( movieData[0].length );
}

