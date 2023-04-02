import std.stdio;

void main(){

    float[] a = [1.0f, 2.0f, 3.0f];
    float[] b = [1.0f, 1.0f, 1.0f];
    float[] c;

    c = b; // This is a POINTER assignment!

    b[1] = 8.0f;
  
    writeln( a ); // [1, 2, 3]
    writeln( b ); // [1, 8, 1]
    writeln( c ); // [1, 8, 1]

    c[1] = 3.0f; // This changes the value we fetch for BOTH, It's a POINTER

    writeln( b ); // [1, 3, 1]

}