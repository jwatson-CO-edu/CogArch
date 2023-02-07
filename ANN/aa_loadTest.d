module aa_loadTest;

import std.stdio;

import utils;

void main(){
    ulong[][] movieData = file_to_dyn_matx_ws!ulong( "../Data/ml-100k/u1.base" );
    writeln( movieData.length ); // 80000 rows
    writeln( movieData[0].length ); // 4 cols
    
}

