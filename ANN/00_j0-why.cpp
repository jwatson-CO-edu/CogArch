#include <iostream>
using std::cout, std::endl, std::flush, std::ostream;

int main(){
    unsigned long int j = 4;
    float k = (j,0); // This is the comma operator
    // https://stackoverflow.com/questions/76133041/assigning-a-tuple-or-mystery-structure-to-a-float#comment134264700_76133041
    cout << "    j = " << j     << endl;
    cout << "(j,0) = " << (j,0) << endl;
    cout << "    k = " << k     << endl;
}
