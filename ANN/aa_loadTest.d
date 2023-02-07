module aa_loadTest;

import std.stdio;

import utils;

void main(){
    ulong[][] movieData = file_to_dyn_matx_ws!ulong( "../Data/ml-100k/u1.base" );
    // FIXME, START HERE: Found file, but FAILED TO LOAD DATA?
    writeln( movieData.length );
    writeln( movieData[0].length );
    writeln( movieData[4000].length );
    writeln( movieData[$-1].length );
}

