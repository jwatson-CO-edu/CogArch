import csv

import numpy as np
from sklearn.decomposition import PCA

rows = 0 
rLen = 0
data = []
Nkp  = 25
pca  = PCA( n_components = Nkp )
# pca  = PCA()

def maxdex():
    # FIXME, START HERE: FIND THE MAX INDEX OF A LIST
    pass

# https://docs.python.org/3/library/csv.html
print( "Reading file ..." ) 

with open( 'seizure-data.csv' ) as csvfile: 
    dataCSV = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate( dataCSV ):
        if i > 0:
            rows += 1
            rLen = len( row[1:] )
            data.append( [float(elem) for elem in row[1:]] )
print( f"There are {rows} rows with {rLen} columns each!  {Nkp} columns will be kept ..." )
print( "Formatting data ..." )
data = np.array( data )
print( "Principal Component Analysis ..." )
pca.fit( data )
print( pca.explained_variance_ratio_ )
print( pca.components_.shape )

# FIXME: FIND THE GREATEST COLUMN IN EACH OF THE COMPONENTS, THESE ARE THE MOST IMPORTANT COLUMNS IN THE DATA
for component in pca.components_:
    pass



        
