import pybullet as p
import time
import pybullet_data

physicsClient = p.connect( p.GUI ) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath( pybullet_data.getDataPath() ) #optionally
p.setGravity( 0, 0, -10 )

planeId = p.loadURDF( "plane.urdf" )
tableId = p.loadURDF( "table/table.urdf", [ 0, 0, 0.0 ] )
cupId   = p.loadURDF( "models/cup.urdf", [ 0, 0, 1.0 ] )

print( type( planeId ), '\n', dir( planeId ) )

# p.resetBasePositionAndOrientation( boxId, startPos, startOrientation )

# startPos = p.Ornp.resetBasePositionAndOrientation( boxId, startPos, startOrientation )

for i in range( 2000 ):
    p.stepSimulation()
    time.sleep( 1.0 / 240.0 )


p.disconnect()