# Cazzolato, Ben. "Derivations of the Dynamics of the Ball and Beam System." School of mechanical engineering, The University of Adelaide 11 (2007): 2010-05.
# https://www.researchgate.net/profile/Ben_Cazzolato/publication/235898231_Derivation_of_the_Dynamics_of_the_Ball_and_Beam_System/links/569338cf08aec14fa55db5a8/Derivation-of-the-Dynamics-of-the-Ball-and-Beam-System.pdf

# fixed parameters from the paper
# L  (m)  length of the beam 
# d  (m)  distance from pivot to the plane of the ball contact on the beam
# D  (m)  distance from the pivot to the center of mass of the beam
# R0 (m)  radius of the ball
# R1 (m)  distance between the axis of rotation of the ball (center of gravity) and the point of contact of the ball with the beam
# m  (kg) mass of the ball
# M  (kg) mass of the beam (located at d (m) from the pivot)
# C1 (N/(m/s)) is the viscous friction coefficient between the ball and the beam (and accounts for the ball rotational viscous losses too since these enter at the same dynamic order)
# C2 is the damping term for the beams velocity
# Jb (kg.m^2)  is the moment of inertia of the beam, including all rotational components such as rotor include beam mass offset (parallel axis theorem)
# g=9.81 (m/s^2) is the gravitational acceleration

# independent variables
# r  (m)   displacement of the ball along the beam. Positive r is when the ball is traveling to the right and r=0 represents the center of the beam
# θ  (rad) is the angular rotation of the beam, where a counter-clockwise direction is Positive

# dependent variables
# τ  (Nm) is the torque the servo-motor applies to the beam, where a counter-closewise directin is Positive

# servo params (not from paper)
# kp  proportional constant 
# servo model: produces a torque corresponding to a reference beam angle u
# τ = kp * (u - θ)
# The beam's friction constant acts as the kd term in a PD controller



# uses the linearized system dynamics from the paper. 
function ballbeam_sim!(state, u, params, dt, sim_steps::Int)
	r, ṙ, θ, θ̇ = state[1], state[2], state[3], state[4]
    
    L, d, D, R0, R1, m, M, C1, C2, Jb, g, kp = params
    
    d² = d^2
    R0² = R0^2
    R1² = R1^2
    
    Δt = dt / sim_steps
    K = 2*R0²*m*d² + 5*R1²*Jb + 2*R0²*Jb
    Q = 5*R1* + 2*R0² + 5 * R1²
    Q2 = (5*R1² + 2 * R0²)
    A21 = -m*g*R1 * Q / K
    A22 = -R1²*C1 * (2*R0² + 5*d² + 10*R1*d + 5*R1² + 5*Jb / m) / K
    A23 = g*R1*(2*R0²*M*D - 5*R1*Jb + 2*R0²*m*d + 5*R1²*M*D + 5*R1*d*M*D) / K
    A24 = -C2*R1*Q / K

    A41 = -m*g*Q2 / K
    A42 = -C1*R1*Q/K
    A43 = g * (2*R0²*m*d + 5*R1²*M*D + 2*R0²*M*D) / K
    A44 = -C2*Q2/K

    B21 = R1 * Q / K
    B41 = Q2 / K


    for i in 1:sim_steps
        τ = kp * (u - θ) #- C2 * θ̇
        dr = ṙ
        dθ = θ̇
        
        dṙ = A21 * r + A22 * ṙ + A23 * θ + A24 * θ̇ + B21 * τ

        # println(dr)
        
        dθ̇ = A41 * r + A42 * ṙ + A43 * θ + A44 * θ̇ + B41 * τ

        r += Δt * dr
        ṙ += Δt * dṙ
        θ += Δt * dθ
        θ̇ += Δt * dθ̇
	end
    # if ball hits the side
    Lhalf = L/2 - R0
    if r > Lhalf
        ṙ = 0.0#0.25 * (Lhalf - r)  # if ball hits the side bounce back with half the velocity (damping by 0.25 so it cannot have infinite energy)
        r = Lhalf
    elseif r < -Lhalf
        ṙ = 0.0#0.25 * (-Lhalf - r)  # if ball hits the side bounce back with half the velocity (damping by 0.25 so it cannot have infinite energy)
        r = -Lhalf
    end

    # if beam tilts past 5 degress it halts (all energy is absorbed)
    θlim = deg2rad(5)
    if θ > θlim
        θ = θlim
        θ̇ = 0.0
    elseif θ < -θlim
        θ = -θlim
        θ̇ = 0.0
    end
	state[1:4] .= r, ṙ, θ, θ̇
	return nothing
end


function ballbeam_params(randomize=false)
    L = 0.25 # 0.25  # range [0.25, 0.5]
    D = 0.0 # torque applied to center of beam
    d = 0.01 
    R0 = 0.008
    R1 = R0  # ball rest ontop of beam
    m = 0.05
    M = 0.25
    C1 = 0.2
    C2 = 3.0  # stable constant perhaps don't change. 
    Jb = (1.0/12.0) * M * L^2 # moment of inertia
    g = 9.81
    kp = 10.0 #8.0  # range [8, 20] # 5 is a pretty slow response

    if randomize
        L = rand() * 0.25 + 0.25    # [0.250,  0.50]
        m = rand() * 0.975 + 0.025  # [0.025,  0.10]
        M = rand() * 0.75 + 0.25    # [0.250,  1.00]
        C1 = rand() * 0.15 + 0.1    # [0.100,  0.25]
        kp = rand() * 12 + 8.0      # [8.000, 20.00]
    end

    return L, d, D, R0, R1, m, M, C1, C2, Jb, g, kp
end


function ballandbeam_fixedgoal(;randomize=false, maxT=20.0, dt=0.05, droptime=true, stochastic_start=true, action_type=:Discrete)
    params = ballbeam_params(randomize)
    
	return create_ballandbeam_finitetime(params, maxT=maxT, dt=dt, droptime=droptime, stochastic_start=stochastic_start, action_type=action_type, goal_mode=:fixed)
end

function ballandbeam_randomgoal(;randomize=false, maxT=20.0, dt=0.05, droptime=true, stochastic_start=true, action_type=:Discrete)
    params = ballbeam_params(randomize)
    
	return create_ballandbeam_finitetime(params, maxT=maxT, dt=dt, droptime=droptime, stochastic_start=stochastic_start, action_type=action_type, goal_mode=:random)
end

function ballandbeam_tacking(;randomize=false, maxT=20.0, dt=0.05, droptime=true, stochastic_start=true, action_type=:Discrete)
    params = ballbeam_params(randomize)
    
	return create_ballandbeam_finitetime(params, maxT=maxT, dt=dt, droptime=droptime, stochastic_start=stochastic_start, action_type=action_type, goal_mode=:track)
end

function create_ballandbeam_finitetime(params; maxT=20.0, dt=0.05, droptime=true, stochastic_start=true, action_type=:Discrete, goal_mode=:fixed)
    X = zeros((5,2))
    L, R0 = params[1], params[4]
    Lhalf = L/2
	X[1,:] .= [-Lhalf+R0, Lhalf-R0]   # ball position range
    X[2,:] .= [-0.7, 0.7]             # ball velocity range (seems about right for fastest ball acceleration)
    θlim = deg2rad(5)
    X[3,:] .= [-θlim, θlim]          # beam is limited to ± 5 degrees
    X[4,:] .= [-2*θlim, 2*θlim]      # this is too high and is probably closer to ±0.1 for default parameters
    X[5,:] .= [-(L - 2*R0), L - 2*R0] # delta to goal position
	S = ([0.0, maxT], X[1:4, :], X[1, :])
    if action_type == :Discrete
        A = 1:3
    else
        A = [-1.0, 1.0]
    end
	sim_steps = 100  # cannot reduce much below here for all configurations of ball and beam. Long beams are easy to simulate. 
    if goal_mode == :track
        d = (L/2 - R0) * 0.9
        y = t->cos(t/2.0+π)*d
        step_fn = (s,a)->ballandbeam_step_track!(s,a,params,y,dt,sim_steps,maxT)
        intfn = ()->bb_sample_initial_track(params, y)
    elseif goal_mode == :fixed
        step_fn = (s,a)->ballandbeam_step!(s,a,params,dt,sim_steps,maxT)
        intfn = ()->bb_sample_initial_fixed(params, stochastic_start)
    elseif goal_mode == :random
        step_fn = (s,a)->ballandbeam_step!(s,a,params,dt,sim_steps,maxT)
        intfn = ()->bb_sample_initial_random(params)
    end
    if droptime
		function get_outcome1(s,a,step_fn)
			t, x, g, r, γ = step_fn(s,a)
			s = (t,x,g)
			return s,x,r,γ
		end
        p = (s,a)->get_outcome1(s,a,step_fn)
        
        function bbd0_obs()
            t,x,g = intfn()
            return (t,x,g), x
        end
        d0 = bbd0_obs
	else
		X = (S[1], X)
		function get_outcome2(s,a,step_fn)
			t, x, g, r, γ = step_fn(s,a)
            s = (t,x,g)
            obs = (t,x)
			return s,obs,r,γ
		end
        p = (s,a)->get_outcome2(s,a,step_fn)
        function bbd0()
            t,x,g = intfn()
            s = (t,x,g)
            obs = (t,x)
            return s,obs
        end
		d0 = bbd0
    end
    
    numT = ceil(Int, maxT / dt)
	meta = Dict{Symbol,Any}()
    meta[:minreward] = -22.21  # max speed and distance from goal on longest beam
    meta[:maxreward] = 0.0
    meta[:minreturn] = -22.21 * numT
    meta[:maxreturn] = 0.0
    meta[:stochastic] = false
    meta[:minhorizon] = numT
    meta[:maxhorizon] = numT
	meta[:discounted] = false
	meta[:episodes] = 200
	render = (state,clearplot=false)->ballbeamplot(state, params)

	m = SequentialProblem(S,X,A,p,d0,meta,render)
end



function bbget_refpos(action::Int)
    θlim = deg2rad(5)
    if action == 1
        return -θlim
    elseif action == 2
        return 0.0
    else
        return θlim
    end
end

function bbget_refpos(action::Float64) 
    return clamp(action, -1.0, 1.0) * deg2rad(5.0)
end



function ballandbeam_step!(state, action, params, dt, sim_steps, maxT)
	u = bbget_refpos(action)
    t,x,g = state
    dcoef = 10.0
    vcoef = 2.0
    olddist = (x[5]*dcoef)^2
    oldv = (x[2]*vcoef)^2
    ballbeam_sim!(x, u, params, dt, sim_steps)
    t += dt
    x[5] = g - x[1]
    curdist = (x[5]*dcoef)^2
    curv = (x[2]*vcoef)^2

    # reward = (olddist - curdist) + (oldv - curv) # shaping reward
    reward = -(curdist + curv)  # negative quadratic cost
	γ = 1.0
	done = t ≥ maxT

	if done
        γ = 0.0
	end
	
	return t, x, g, reward, γ
end



function ballandbeam_step_track!(state, action, params, y, dt, sim_steps, maxT)
	u = bbget_refpos(action)
    t,x,g = state
    dcoef = 10.0
    olddist = (x[5]*dcoef)^2
    ballbeam_sim!(x, u, params, dt, sim_steps)
    t += dt
    g = y(t)
    x[5] = g - x[1]
    curdist = (x[5]*dcoef)^2

    reward = -curdist
	γ = 1.0
	done = t ≥ maxT

	if done
        γ = 0.0
	end
	
	return t, x, g, reward, γ
end



function bb_sample_initial_fixed(params, stochastic_start=true)
    x = zeros(5)
    L, R = params[1], params[4]
    x[1] = -L/2 + R
    
    if stochastic_start
        x[1] += rand() * 0.2 * L
    end
    g = L/4
    x[5] = g - x[1]

	return 0.0, x, g
end

function bb_sample_initial_random(params)
    x = zeros(5)
    L, R = params[1], params[4]
    Lhalf = L/2 - R
    
    
    g = rand() * 2*Lhalf - Lhalf
    if g < 0
        x[1] = rand() * 0.5 * Lhalf
    else
        x[1] = -rand() * 0.5 * Lhalf 
    end

    x[5] = g - x[1]

	return 0.0, x, g
end

function bb_sample_initial_track(params, y)
    x = zeros(5)
    L, R = params[1], params[4]
    x[1] = -L/2 + R
    t = 0.0
    g = y(t)
    x[5] = g - x[1]

	return t, x, g
end

# Position of the ball
function bb_ballpos(r, θ, d)
    x1, y1 = r * cos(θ), r * sin(θ)
    x2, y2 = d * cos(θ+π/2), d * sin(θ + π/2)
    return x1+x2, y1+y2
end

@userplot BallBeamPlot
@recipe function f(ap::BallBeamPlot)
    state, params = ap.args
    t,x,g = state
    r, θ = x[1], x[3]
    L, d, D, R0= params[1:4]
    # println("R0 $R0")
	maxlen = (L/2)*1.2
    dlow = -d
    
    ballx, bally = bb_ballpos(r, θ, d + R0)
    goalx, goaly = bb_ballpos(g, θ, d + R0)
    
    # println(ballx, " ", bally)
	legend := false
	xlims := (-maxlen, maxlen)
	ylims := (-maxlen*sin(deg2rad(5))-(2*R0+d), maxlen*sin(deg2rad(5))+(2*R0+d))
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	
    
    # beam
	@series begin
        seriescolor := :orange
        linecolor := nothing
        seriestype := :shape
		beamx11, beamy11 = bb_ballpos(-L/2, θ, d)
        beamx21, beamy21 = bb_ballpos(L/2, θ, d)
        beamx22, beamy22 = bb_ballpos(L/2, θ, dlow)
        beamx12, beamy12 = bb_ballpos(-L/2, θ, dlow)
		[beamx11, beamx21, beamx22, beamx12], [beamy11, beamy21, beamy22, beamy12]
    end
    
    # left end
	@series begin
        seriescolor := :orange
        linecolor := nothing
        seriestype := :shape
		beamx11, beamy11 = bb_ballpos(-L/2 - d, θ, d+2*R0)
        beamx21, beamy21 = bb_ballpos(-L/2, θ, d+2*R0)
        beamx22, beamy22 = bb_ballpos(-L/2, θ, dlow)
        beamx12, beamy12 = bb_ballpos(-L/2 -d, θ, dlow)
		[beamx11, beamx21, beamx22, beamx12], [beamy11, beamy21, beamy22, beamy12]
    end

    # right end
	@series begin
        seriescolor := :orange
        linecolor := nothing
        seriestype := :shape
		beamx11, beamy11 = bb_ballpos(L/2, θ, d+2*R0)
        beamx21, beamy21 = bb_ballpos(L/2 + d, θ, d+2*R0)
        beamx22, beamy22 = bb_ballpos(L/2 + d, θ, dlow)
        beamx12, beamy12 = bb_ballpos(L/2, θ, dlow)
		[beamx11, beamx21, beamx22, beamx12], [beamy11, beamy21, beamy22, beamy12]
    end

	# goal location
	@series begin
		seriestype := :shape
		linecolor := nothing
        seriescolor := :green
        seriescolor := :green
		aspect_ratio := 1.
		fillalpha := 0.4

		circle_shape(goalx, goaly, R0)
	end

    # ball location
	@series begin
		seriestype := :shape
		linecolor := nothing
		seriescolor := :red
        aspect_ratio := 1.
        
		circle_shape(ballx, bally, R0)
    end

    
end