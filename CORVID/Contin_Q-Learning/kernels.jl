########## INIT ####################################################################################

using LinearAlgebra
using NearestNeighbors
using StaticArrays
include("utils.jl")



########## PAPER 1 #################################################################################


##### k Nearest Neighbors ########################

"""
Return the indices and weights of the `k` nearest neighbors
NOTE: This function assumes that you have already built a kD-Tree and cached it in the `QStore`
"""
function K_kNN( kdTree, q; k = 6 )
    factor      = 1.0/k
    idxs, dists = knn( kdTree, q, k )
    return idxs, repeat( [factor], k )
end


"""
Use k Nearest Neighbors of `q` to retrive value from a `QStore`
"""
function query_value_w_knn( kdTree, pnts, vals, q; k = 6 )
    idxs, wgts = K_kNN( kdTree, q; k = k )
    p_  , v_   = fetch_by_indices( pnts, vals, idxs )
    return val_weighted_avg( v_, wgts )
end


##### Fixed Radius ###############################

"""
Return the indices and weights of the neighbors within `rad` of `d`
NOTE: This function assumes that you have already built a Ball Tree and cached it in the `QStore`
"""
function K_fixed_rad( blTree, q; rad = 1.0 )
    idxs     = inrange( blTree, q, rad )
    n        = size( idxs, 1 )
    factor   = 1.0/n
    return idxs, repeat( [factor], n )
end


"""
Use a ball within `radius` of `q` to retrive value from a `QStore`
"""
function query_store_w_radius( blTree, pnts, vals, q; radius = 1.0 )
    idxs, wgts = K_fixed_rad( blTree, q; rad = radius )
    p_  , v_   = fetch_by_indices( pnts, vals, idxs )
    return val_weighted_avg( v_, wgts )
end


##### (Truncated) Gaussian Kernel ################


"""
Used by the Gaussian Kernel
"""
function gaussian( s )
    exp( -s^2 / 2.0 )
end


"""
Return the indices and weights of the neighbors chosen by the Truncated Gaussian Function
"""
function K_trunc_Gauss( blTree, pnts, vals, q; scale = 1.0, lim = 1.0 )
    # Init
    rtnDex = [] # -------- Index associated with each weight
    rtnWgt = [] # -------- The weight of each index
    totl   = 0.0 # ------- Total of all distance calcs
    dScl   = scale * lim # Search radius
    
    # Search within the truncation hypersphere
    idxs, wgts = K_fixed_rad( blTree, q; rad = dScl )
    p_  , v_   = fetch_by_indices( pnts, vals, idxs )
    nPts       = size( p_, 2 ) # Total number of points/anchors
    
    # For each anchor point in our results
    for j in 1:nPts
        pnt = p_[:,j] # Fetch point
        idx = idxs[j]
        s_j = norm(q-pnt)/scale #- Compute the distance from the query
        # If the distance is equal to or less than the distance limit, then consider an applicable point
        g_j = gaussian( s_j ) #- Compute Gaussian distance
        append!( rtnDex, idx ) # Store index
        append!( rtnWgt, g_j ) # Store distance
        totl += g_j # ---------- Sum distance
    end
    rtnWgt ./= totl # Normalize the weights (MUST sum to 1.0)
    return rtnDex, rtnWgt # Return applicable points and their weights
end


"""
Use a Gaussian kernel centered on `q` to retrive value from a `QStore`
"""
function query_value_w_gauss( blTree, pnts, vals, q; scale = 1.0, lim = 1.0 )
    idxs, wgts = K_trunc_Gauss( blTree, pnts, vals, q; scale = scale, lim = lim )
    p_  , v_   = fetch_by_indices( pnts, vals, idxs )
    return val_weighted_avg( v_, wgts )
end



########## PAPER 2 #################################################################################

"""
Each point contributes the inverse of its manhattan distance from the interpolated point
"""
function manhatt_contrib( vec )
    return 1.0 / manhattan_norm( vec )
end


"""
Return weights for `pnts` according to the closeness to `q`
"""
function manhatt_contrib_to_q( pnts, q )
    Npts  = size( pnts, 2 )
    wghts = zeros( Npts )
    for j = 1:Npts
        wghts[j] = manhatt_contrib( q - pnts[:,j] )
    end
    return vec_norm( wghts )
end


"""
Use k Nearest Neighbors of `q` to retrive value from a `QStore`
"""
function query_value_fuzzy( kdTree, pnts, vals, q; k = 3 )
    idxs, wgts = K_kNN( kdTree, q; k = k )
    if !(-1 in idxs)
        p_  , v_  = fetch_by_indices( pnts, vals, idxs )
        wgts      =  manhatt_contrib_to_q( p_, q )
        # println( "query_value_fuzzy - weights: ", wgts, ", sum: ", sum( wgts ) ) # Sums to 1 as intended
        return val_weighted_avg( v_, wgts )
    else
        return 0.0
    end
end


"""
Return indices of k Nearest Neighbors of `q` and the degree to which to blend with them
"""
function query_contrib_to_neighbors( kdTree, pnts, vals, q; k = 3 )
    idxs, w_ = K_kNN( kdTree, q; k = k )
    p_  , v_ = fetch_by_indices( pnts, vals, idxs )
    wgts     = manhatt_contrib_to_q( p_, q )
    return idxs, wgts
end


##### Exp Version ################################

"""
Each point contributes the inverse exponent of its manhattan distance from the interpolated point
"""
function manhatt_contrib_exp( vec )
    return 1.0 / exp( manhattan_norm( vec ) )
end


"""
Return weights for `pnts` according to the closeness to `q`
"""
function manhatt_contrib_to_q_exp( pnts, q )
    Npts  = size( pnts, 2 )
    wghts = zeros( Npts )
    for j = 1:Npts
        wghts[j] = manhatt_contrib_exp( q - pnts[:,j] )
    end
    return vec_norm( wghts )
end


"""
Use k Nearest Neighbors of `q` to retrive value from a `QStore`
"""
function query_value_fuzzy_exp( kdTree, pnts, vals, q; k = 3 )
    idxs, wgts = K_kNN( kdTree, q; k = k )
    if !(-1 in idxs)
        p_  , v_  = fetch_by_indices( pnts, vals, idxs )
        wgts      =  manhatt_contrib_to_q_exp( p_, q )
        # println( "query_value_fuzzy_exp - weights: ", wgts, ", sum: ", sum( wgts ) ) # Sums to 1 as intended
        return val_weighted_avg( v_, wgts )
    else
        return 0.0
    end
end


"""
Return indices of k Nearest Neighbors of `q` and the degree to which to blend with them
"""
function query_contrib_to_neighbors_exp( kdTree, pnts, vals, q; k = 3 )
    idxs, w_ = K_kNN( kdTree, q; k = k )
    p_  , v_ = fetch_by_indices( pnts, vals, idxs )
    wgts     = manhatt_contrib_to_q_exp( p_, q )
    return idxs, wgts
end