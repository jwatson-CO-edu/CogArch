import numpy as np
from uuid import uuid4

########## BASE CLASSES ############################################################################

class Symbol:
    """ Base class for all PLOVER datatypes """
    def __init__( self, typeName = "Symbol", label = "", representation = None ):
        self.ID    = uuid4()
        self.type  = typeName
        self.label = label
        self.rep   = representation


class Object( Symbol ):
    """ Base class for objects with a pose in 3D space """
    def __init__( self, typeName = "Symbol", label = "", representation = None, pose = None ):
        super().__init__( typeName, label, representation )
        self.pose    = pose if (pose is not None) else np.eye(4)
        self.poseAbs = np.eye(4)


class Body( Object ):
    """ Base class for `Object`s with a 3D extent """
    def __init__( self, typeName = "Body", label = "", mesh = None, pose = None ):
        super().__init__( typeName, label, mesh, pose )
