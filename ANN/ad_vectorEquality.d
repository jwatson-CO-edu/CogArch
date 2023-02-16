import std.stdio;

void main(){

    float[] a = [1.0f, 2.0f, 3.0f];
    float[] b = [1.0f, 1.0f, 1.0f];
    float[] c = [1.0f, 2.0f, 3.0f];

    writeln( a == b );
    writeln( a == c );

    writeln( a[0..$-1] );

    writeln( a[0..$-1] == b[0..$-1] );
    writeln( a[0..$-1] == c[0..$-1] );

}