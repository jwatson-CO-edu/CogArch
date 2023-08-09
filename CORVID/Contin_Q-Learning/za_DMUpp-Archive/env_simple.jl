########## SIMPLEST ENVIRONMENT ####################################################################

"""
Assemble <State,Action> into Q-state
"""
function get_Q( X, A )
    res = zeros( nQ );
    res[ 1:nX ]    = X;
    res[ nX+1:nQ ] = A;
    return res;
end


"""
Disassemble <State,Action> into Q-state
"""
function XA_from_Q( Q )
    return Q[ 1:nX ], Q[ nX+1:nQ ];
end

    
"""
Extremely simple dynamics, effort is the position offset applied
"""
function dyn( X, A )
    A = clamp_vec( A, _A_DOMAIN );
    X = clamp_vec( X + A, _X_DOMAIN );
    return get_Q( X, A );
end