########## CONSTANTS ###############################################################################

# https://github.com/AadityaPatanjali/gym-cartpolemod
# magnitude of the force = 10.0
g    = 9.81
m_c  = 1.0
m_p  = 0.1
l    = 0.5
mu_c = 5e-4/4.0
mu_p = 2e-6/4.0



########## DYNAMICS ################################################################################


"""
Return the normal force of the cart on the track
"""
function N_cart( thetaDotDot, thetaDot, theta )
    return (m_c + m_p)*g - m_p*l*( thetaDotDot*sin(theta) + (thetaDot^2)*cos(theta) )
end


"""
Return the angular acceleration of the pole
"""
function angular_accel( F, N_c, thetaDot, theta, xDot )
    
    pt1 = mu_c*sign( N_c*xDot )
    pt2 = (-F - m_p*l*(thetaDot^2)*( sin(theta) + pt1*cos(theta) ) )/(m_c + m_p) + g*pt1
    num = g*sin(theta) + cos(theta)*pt2 - ( mu_p*thetaDot )/( m_p*l )
    den = l*( 4/3 - (( m_p*cos(theta) )/( m_c + m_p ))*( cos(theta) - pt1 ) )
        
    return num / den
end


"""
Return the linear acceleration of the cart
"""
function linear_accel( F, N_c, thetaDotDot, thetaDot, theta, xDot )
    return ( F + m_p*l*( (thetaDot^2)*sin(theta) - thetaDotDot*cos(theta) ) - mu_c*N_c*sign( N_c*xDot ) )/( m_c + m_p )
end


"""
Calculate the next state according to "Correct equations for the dynamics of the cart-pole system"
"""
function cartpole_dyn( X, A, ts )
    
    # 1. Unpack state
    thetaDotDot = X[1] # Angular acceleration
    thetaDot    = X[2] # Angular velocity
    theta       = X[3] # Angle
    xDotDot     = X[4] # Linear acceleration
    xDot        = X[5] # Linear velocity
    x           = X[6] # Fulcrum position
    N_c         = X[7] # Normal force of the cart on the track
    
    # 2. Unpack action
    F = A
    
    # 3. Calc angular accel
    # "we may assume that N_c has the same sign as at the previous timestep and compute thetaDotDot"
    thetaDotDot_tp1 = angular_accel( F, N_c, thetaDot, theta, xDot )
    
    # 4. Calc normal force
    # "We then compute N_c using the value of thetaDotDot that we obtained"
    N_c_tp1 = N_cart( thetaDotDot_tp1, thetaDot, theta )
    
    # 5. "If N_c changes sign, we compute again thetaDotDot taking into account the new sign"
    if sign( N_c ) != sign( N_c_tp1 )
        thetaDotDot_tp1 = angular_accel( F, N_c_tp1, thetaDot, theta, xDot )
    end
    
    # 5. "Finally, we compute xDotDot"
    xDotDot_tp1 = linear_accel( F, N_c_tp1, thetaDotDot, thetaDot, theta, xDot )
    
    # 6. Integrate angle with the trapezoid rule
    thetaDot_tp1 = thetaDot + ((thetaDotDot + thetaDotDot_tp1)/2) * ts # prev vel + change in vel
    theta_tp1    = theta    + ((thetaDot    + thetaDot_tp1   )/2) * ts # prev pos + change in pos
    
    # 6. Integrate position with the trapezoid rule
    xDot_tp1 = xDot + ((xDotDot + xDotDot_tp1)/2) * ts # prev vel + change in vel
    x_tp1    = x    + ((xDot    + xDot_tp1   )/2) * ts # prev pos + change in pos
    
    # N. Package the state vector and return it
    return [ thetaDotDot_tp1, thetaDot_tp1, theta_tp1, xDotDot_tp1, xDot_tp1, x_tp1, N_c_tp1 ]
end