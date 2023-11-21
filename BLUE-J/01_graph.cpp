////////// INIT ////////////////////////////////////////////////////////////////////////////////////

#include <string>
using std::string;
#include <map>
using std::map;
#include <memory>
using std::shared_ptr;
#include <vector>
using std::vector;

typedef map<string,float> symdist;

////////// UTILS ///////////////////////////////////////////////////////////////////////////////////

// FIXME, START HERE: NEED A MEANS OF GENERATING GUARANTEED UNIQUE STRINGS

////////////////////////////////////////////////////////////////////////////////////////////////////
// NAMESPACE: BLUE_J                                                                              //
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace BLUE_J{

class UniqueCognitiveStruct{ public:
    string  symbol; // Guaranteed unique string
    string  type; //-- Flexible type designator // Might this also be a `symdist`?
    symdist labels; // Possible semantic labels this object *might* have
    float   weight; // How much we care about this particular object
};   

class Node;  typedef shared_ptr<Node> nodePtr;
class Edge;  typedef shared_ptr<Edge> edgePtr;

class Node : public UniqueCognitiveStruct{ public:
    vector<edgePtr> outgoing;
};

class Edge : public UniqueCognitiveStruct{ public:
    nodePtr head;
    nodePtr tail;    
};

};

