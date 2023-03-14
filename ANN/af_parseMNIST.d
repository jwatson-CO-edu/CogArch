// rdmd af_parseMNIST.d

import std.stdio; // -------------- `writeln`
import std.file; // -------------- `read`
import std.conv;
import std.bitmanip;

void main(){
    // File trainImageFile = File( "../Data/MNIST/train-images.idx3-ubyte", "r" );

    // writeln(  cast(int) read( "../Data/MNIST/train-images.idx3-ubyte", 4 ) );
    // writeln(  read( "../Data/MNIST/train-images.idx3-ubyte", 4 ) );

    ubyte[] buffer = cast(ubyte[]) read( "../Data/MNIST/train-images.idx3-ubyte", 4 );
    
    // writeln( peek!(int, Endian.littleEndian)(buffer) );
    writeln( peek!(int, Endian.bigEndian)(buffer) ); // 0x00000803(2051) magic number

}