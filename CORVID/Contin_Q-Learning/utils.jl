########## INIT ####################################################################################

using LinearAlgebra
using NearestNeighbors
using StaticArrays
using Luxor



########## RECTANGULAR GRIDS #######################################################################
"""
Get all possible combos of `tickLists` elements
"""
function regular_grid_helper( tickLists, i = 1, curVec = nothing, allPts = nothing )
    mDim = size( tickLists, 1 )
    
    # If this is the root call, then create the return matrix
    if allPts == nothing
        nPts = 1
        for tcs in tickLists
            nPts *= size( tcs, 1 )
        end
        allPts = zeros(mDim, nPts)
    end
    
    # If we are beginning a new vector
    if curVec == nothing
        curVec = []
    end
    
    # Base Case: Last Tick Dimension
    if mDim  == 1
        for tick in tickLists[1]
            nuVec       =  vcat(curVec, tick) 
            allPts[:,i] =  nuVec
            i += 1
        end
        curVec = nothing
    # Recursive Case
    else
        for tick in tickLists[1]
            _, i = regular_grid_helper( tickLists[2:end], i , vcat(curVec,tick), allPts )
        end
    end
    
    return allPts, i
end


"""
Return per-dim ticks of a regular grid of points across `rangeMatx` and `spacingArr` per-dim 
"""
function get_nD_ticks( rangeMatx, spacingArr )
    rtnTicks = []
    mDim     = size( rangeMatx, 1 )
    
    # 1. Generate tick marks in each dimension
    for j in 1:mDim 
        dimTicks = []
        bgn      = rangeMatx[j,1]
        stp      = rangeMatx[j,2]
        k        = 1
        append!( dimTicks, bgn )
        while true
            t_k = bgn + (k * spacingArr[j])
            if t_k <= stp
                append!( dimTicks, t_k )
            else
                break
            end
            k += 1
        end
        append!( rtnTicks, [dimTicks] )
    end
    return rtnTicks
end


"""
Return a regular grid of points across `rangeMatx` and per-dim space according to `spacingArr`
"""
function regular_grid_pts_nD( rangeMatx, spacingArr )
    tickLists = get_nD_ticks( rangeMatx, spacingArr )
    arr, _ = regular_grid_helper( tickLists )
    return arr
end



########## ACCESS & INTERP #########################################################################

"""
Return the anchors and values associated with the `idxList`
"""
function fetch_by_indices( pnts, vals, idxList )
    N = size( idxList, 1 ) # -------- How many queries are there?
    p = zeros( size( pnts, 1 ), N ) # Allocate points
    v = zeros( N ) # ---------------- Allocate values
    for i = 1:N
        if (i>0)
            j = idxList[i]
            p[:,i] = pnts[:,j]
            v[i]   = vals[j]
        end
    end
    return p, v
end


"""
Compute a weighted average value from `vals`, `wgts`
NOTE: This function assumes that the scalar elements of `wgts` sum to 1.0
"""
function val_weighted_avg( vals, wgts )
    nPts = size( vals, 1 )
    r    = 0.0
    for j in 1:nPts
        v = vals[j]
        w = wgts[j]
        r += (w * v)
    end
    return r
end


"""
Calc the unit vector in the same direction as arg
"""
function vec_unit( vec )
    len = norm( vec )
    if len == 0.0
        return vec
    else
        return vec / len
    end
end


"""
Compute a weighted average value from `vals`, `wgts`, but only for positive `vals` 
NOTE: This function assumes that the scalar elements of `wgts` sum to 1.0
"""
function val_weighted_avg_pos( vals, wgts )
    nPts = size( vals, 1 )
    r    = 0.0
    wgtP = wgts[:]
    for j in 1:nPts
        v = vals[j]
        if v < 0.0
            wgtP[j] = 0.0
        end
    end
    wgtP = vec_unit( wgtP )
    for j in 1:nPts
        v = vals[j]
        w = wgts[j]
        r += (w * v)
    end
    return r
end



########## TESTING FUNCTIONS #######################################################################


"""
Use `truthFunc` to load the correct values for each anchor in `qSto`
NOTE: This function assumes that `truthFunc` has the same `mDim` as `truthFunc`
"""
function assoc_pnts_with_true_values( pnts, vals, truthFunc )
    # For every column of stored points, compute the truth value
    nPts = size( pnts, 2 )
    for j in 1:nPts
        vals[j] = truthFunc( pnts[:,j] )
    end
end


"""
Hyperplane with a uniform positive slope along all axes
"""
function get_nD_example_plane_eq( dims, slope )
    factor = slope / dims
    Y = repeat( [factor], dims )
    return function( X )
        dot( X, Y )
    end
end


"""
Cone of `height` that slopes downward from origin 
"""
function get_nD_example_cone_eq( height, slope )
    return function( X )
        height - norm( X ) * slope
    end
end


"""
Hill of `height` that slopes downward from origin 
"""
function get_nD_example_hill_eq( height, scale )
    return function( X )
        height * exp( -norm(X)/scale )
    end
end


"""
Return a function that rewards increase in height
"""
function get_planar_reward_function( dim, slope )
    heightEQ = get_nD_example_plane_eq( dim, slope );
    return function ( s, a, sP )
        # return heightEQ( sP ) - heightEQ( s ) # This is not consistent if you are taking the gradient of a wide area with sparse visits
        return heightEQ( sP ) # The Reward has to be right!
    end
end



########## UNIFORM SAMPLING ########################################################################

"""
Sample from a uniform distribution [ rMin , rMax )
"""
function sample_uniform_real( rMin , rMax )
    span = abs( rMax - rMin )
    return span*rand() + rMin
end


"""
Sample uniformly from a hypercuboid `bbox`
"""
function sample_bbox_uniform( bbox )
    mDim   = size( bbox, 1 )
    rtnArr = zeros( mDim )
    # print( mDim, ' ', rtnArr )
    # display( bbox )
    for i = 1:mDim
        rMin = bbox[i,1]
        # println( "rMin:", rMin )
        rMax = bbox[i,2]
        # println( "rMax:", rMax )
        rtnArr[i] = sample_uniform_real( rMin , rMax )
    end
    return rtnArr
end


"""
Draw `N` samples uniformly from a hypercuboid `bbox`
"""
function sample_bbox_N( bbox, N )
    mDim   = size( bbox, 1 )
    rtnArr = zeros( mDim, N )
    for j = 1:N
        rtnArr[:,j] = sample_bbox_uniform( bbox )
    end
    return rtnArr
end



########## VECTORS & LINEAR ALGEBRA ################################################################

"""
Clamp a vector [..., v_i, ...] to column matrix [ ..., [v_{i,lo}, v_{i,hi}], ... ]
Accept multiple arg types: https://stackoverflow.com/a/52394797
"""
function clamp_vec( x::Union{Vector, Array}, dom::Matrix{ Float64 } )
    mDim   = max(size(x)...)
    rtnVec = zeros( mDim )
    for i = 1:mDim
        rtnVec[i] = clamp( x[i], dom[i,1], dom[i,2] )
    end
    return rtnVec
end





##### Manhattan Distance #########################

"""
Compute the Manhattan Distance from `vec` to the origin
"""
function manhattan_norm( vec )
    return sum( abs.( vec ) )
end


"""
Calc version of `vec` normalized to sum to 1.0, The contributions of all neighbors must sum to 1.0
"""
function vec_norm( vec )
    len = sum( vec )
    if (len == 0.0) #|| (len == -0.0)
        return vec
    else
        return vec / len
    end
end



########## CARTPOLE EVAL ###########################################################################

"""
Return number of seconds that penulum was within double-sided `angleMargin` of vertical
"""
function vertical_score_s( stateHistory, angleMargin, ts )
    angles = stateHistory[3,:]
    N      = length( angles )
    score  = 0.0
    for j = 1:N
        if abs( angles[j] ) <= angleMargin
            score += ts
        end
    end
    return score
end


"""
Create an animated GIF of a cartpole state sequence
"""
function render_cartpole_episode( Xs, ts )
    
    # 0. Init
    FPS  = 30
    T    = size( Xs, 2 )
    skip = Int64( floor((1/FPS)/ts) )
    frms = Int64( ceil(T/skip) )

    # 1. Background func
    function backdrop(scene, framenumber)
        background("grey50")
    end

    # 2. Per frame func
    function frame( scene, framenumber )

        # Extract relevant state
        theta = Xs[3,(framenumber-1)*skip+1]
        x     = Xs[6,(framenumber-1)*skip+1]/4
        
        # Track
        sethue("blue")
        rule(Point(0, 0), 0.0)

        # Cart
        sethue("black")
        box(Point(x, 0), 70,  40, action = :fill)

        # Pole
        sethue("white")
        Luxor.translate( x, 0 )
        Luxor.translate( @polar (-100, theta+pi/2.0) )
        Luxor.rotate( theta )
        box(Point(0, 0), 15, 200, action=:fill)

    end

    # 3. Instantiate movie
    demo = Movie(800, 800, "test")

    # 4. Populate movie
    Luxor.animate( demo, [
        Scene(demo, backdrop, 1:frms),
        Scene(demo, frame, 1:frms)
    ], creategif=true)
end
