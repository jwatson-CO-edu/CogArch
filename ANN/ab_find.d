// Search for an element in a dynamic array

import std.stdio; // ------------- `writeln`
import std.algorithm.searching; // `countUntil`, linear search

void main(){

    uint[] num;

    for( uint i = 0; i < 10; i++ ){
        num ~= i;
    }
    ulong founDex = num.countUntil!( c => c == 5 );
    writeln( founDex );
    writeln( num[ founDex ] );

}