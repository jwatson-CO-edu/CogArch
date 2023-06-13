import csv

import numpy as np
from sklearn.decomposition import PCA

rows = 0 
rLen = 0
data = []
Nkp  = 25
pca  = PCA( n_components = Nkp )
# pca  = PCA()

def maxdex( lst ):
    """ Find the first index of the maximum value in the list """
    valMax = max( lst )
    return lst.index( valMax )

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

print( "Principal Components:" )
sigCols = []
reData  = np.zeros( (data.shape[0],Nkp,) )
for component in pca.components_:
    abLst = [abs( elem ) for elem in component]
    mxdx = maxdex( abLst )
    print( mxdx, ",", max( abLst ) )
    sigCols.append( mxdx )

for i, col in enumerate( sigCols ):
    reData[:,i] = data[:,col]

print( reData[0,:] )

nuFileName = 'seizure-reduced-25col_no-headings.csv'
with open( nuFileName, 'w') as csvfile:
    spamwriter = csv.writer( csvfile, delimiter=',' )
    for row in reData:
        spamwriter.writerow( row )

print( f"Wrote {nuFileName} with {reData.shape[0]} rows and {reData.shape[1]} columns!" )




        
