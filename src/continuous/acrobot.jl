struct AcrobotParams{T} <:Any where {T<:Real}
    m1::T 	# mass of first link
    m2::T 	# mass of second link
    l1::T 	# length of first link
	l2::T 	# length of second link
	lc1::T 	# position of center mass of link1
	lc2::T 	# position of center mass of link2
	i1::T  	# link1 moment of inertia
	i2::T  	# link2 moment of inertia
	g::T   	# gravity force (not directional)
	fmag::T # max force magnitude

    AcrobotParams() = new{Float64}(1., 1., 1., 1., 0.5, 0.5, 1., 1., 9.8, 1.)
    AcrobotParams(T::Type) = new{T}(1., 1., 1., 1., 0.5, 0.5, 1., 1., 9.8, 1.)
	AcrobotParams(T::Type, m1, m2, l1, l2, g) = new{T}(m1, m2, l1, l2, 0.5*l1, 0.5*l2, 1., 1., g, 1.)
end

function acrobot_finitetime(;randomize=false, maxT=400.0, dt=0.2, Atype=:Discrete, droptime=true, stochastic_start=false)
	if randomize
		params = random_acrobot_params()
	else
		params = AcrobotParams()
	end
	return create_finitetime_acrobot(params; maxT=maxT, dt=dt, Atype=Atype, droptime=droptime, stochastic_start=stochastic_start)
end

function create_finitetime_acrobot(params::AcrobotParams; maxT=400.0, dt=0.2, Atype=:Discrete, droptime=true, stochastic_start=false)
    X = zeros((4,2))
    X[1,:] .= [-π, π]               # theta1 range
	X[2,:] .= [-π, π]               # theta2 range
	X[3,:] .= [-4. * π, 4. * π]     # theta1Dot range
	X[4,:] .= [-9. * π, 9. * π]     # theta2Dot range
    S = ([0. maxT],				# time range
		X)           	# thetaDot range
						
	if Atype==:Discrete
		A = 1:3
	else
		A = [-1.0 1.0]
    end
    
	if droptime
		X = S[2]
		function get_outcome1(s,a,params,dt,maxT,stochastic_start)
			t, x, r, γ = acrobot_step!(s,a, params, dt, maxT, stochastic_start)
			s = (t,x)
			return s,x,r,γ
		end
        p = (s,a)->get_outcome1(s,a,params,dt,maxT,stochastic_start)
        
        function abd0_obs()
            t,x = acrobot_sample_initial(stochastic_start)
            return (t,x), x
        end
        d0 = abd0_obs
	else
		X = S
		function get_outcome2(s,a,params,dt,maxT,stochastic_start)
			t, x, r, γ = acrobot_step!(s,a, params, dt, maxT,stochastic_start)
			s = (t,x)
			return s,s,r,γ
		end
        p = (s,a)->get_outcome2(s,a,params,dt,maxT,stochastic_start)
        function abd0()
            s = acrobot_sample_initial(stochastic_start)
            return s,s
        end
        d0 = abd0
    end
    
    
	meta = Dict{Symbol,Any}()
    meta[:minreward] = -0.1
    meta[:maxreward] = 0.0
    meta[:minreturn] = -ceil(Int, maxT / dt)
    meta[:maxreturn] = -0.9
    meta[:stochastic] = false
    meta[:minhorizon] = 10
    meta[:maxhorizon] = ceil(Int, maxT / dt)
	meta[:discounted] = false
	meta[:episodes] = 400
	meta[:threshold] = -10.0
	
	render = (state,clearplot=false)->acrobotplot(state, params)

	m = SequentialProblem(S,X,A,p,d0,meta,render)
	return m
end

function rk_helper(s, tau, params::AcrobotParams)
	m1, m2, l1, l2, lc1, lc2, i1, i2, g = params.m1, params.m2, params.l1, params.l2, params.lc1, params.lc2, params.i1, params.i2, params.g

	d1 = m1*lc1^2 + m2*(l1^2 + lc2^2 + 2. * l1*lc2*cos(s[2])) + i1 + i2
	d2 = m2*(lc2^2 + l1*lc2*cos(s[2])) + i2

	phi2 = m2*lc2*g*cos(s[1] + s[2] - (π / 2.))
	phi1 = (-m2*l1*lc2*s[4]^2 * sin(s[2]) - 2. * m2*l1*lc2*s[4]*s[3] * sin(s[2]) + (m1*lc1 + m2*l1)*g*cos(s[1] - (π / 2.)) + phi2)

	newa2 = ((1. / (m2*lc2^2 + i2 - (d2^2) / d1)) * (tau + (d2 / d1)*phi1 - m2*l1*lc2*s[3]^2 * sin(s[2]) - phi2))
	newa1 = ((-1. / d1) * (d2*newa2 + phi1))
	return @SVector [s[3], s[4], newa1, newa2]
end

function acrobot_sim!(state::Array{T,1}, u::T, params::AcrobotParams{T}, dt) where {T}
	theta1, theta2, theta1Dot, theta2Dot = state
	hilf = @SVector [theta1, theta2, theta1Dot, theta2Dot] #deepcopy(state)

	h = dt / 10.0

	for i in 1:10
		s0_dot = rk_helper(hilf, u, params)
		s1 = hilf + (h / 2.) * s0_dot

		s1_dot = rk_helper(s1, u, params)
		s2 = hilf + (h / 2.) * s1_dot

		s2_dot = rk_helper(s2, u, params)
		s3 = hilf + (h / 2.) * s2_dot

		s3_dot = rk_helper(s3, u, params)
		hilf = hilf + (h / 6.) * (s0_dot + 2. * (s1_dot + s2_dot) + s3_dot)
	end

	theta1 = mod(hilf[1] + π, 2. * π) - π
	theta2 = mod(hilf[2] + π, 2. * π) - π

	theta1Dot = clamp(hilf[3], -4. * π, 4. * π)
	theta2Dot = clamp(hilf[4], -9. * π, 9. * π)

	state .= theta1, theta2, theta1Dot, theta2Dot

end

function get_torque(params::AcrobotParams, action::Int)
    if action <= 0 || action > 3
        error("Action needs to be an integer in [1, 3]")
    end
    u = 0.0
    u = (Float64(action) - 2.0) * params.fmag
    return u
end

function get_torque(params::AcrobotParams, action::Float64)
    u = clamp(action, -1., 1.) * params.fmag
    return u
end

function acrobot_sample_initial(stochastic_start)
    t = 0.0
    x = zeros(4)
    if stochastic_start
        x[1:2] .= rand(Uniform(-π * 5. / 180., π * 5. / 180.), 2)
    end
    return t, x
end
function acrobot_step!(state, action, params, dt, maxT, stochastic_start)
    u = get_torque(params, action)
    t, x = state
	acrobot_sim!(x, u, params, dt)
	t += dt

    reward = -0.1
    γ = 1.0
	done = acrobot_terminal(t, x, params, maxT)
	if done
        reward = 0.
        γ = 0.0
        t, x = acrobot_sample_initial(stochastic_start)
	end

	return t, x, reward, γ
end

function acrobot_terminal(t, state, params::AcrobotParams, maxT)
	elbowY = -params.l1*cos(state[1])
	handY = elbowY - params.l2*cos(state[1] + state[2])
	anglecond = handY > params.l1
	timecond = t ≥ maxT
    done = anglecond | timecond
    
	return done
end


@userplot AcrobotPlot
@recipe function f(ap::AcrobotPlot)#, state, params::AcrobotParams)
# @recipe function f(params::AcrobotParams, state)
	state, params = ap.args
	t,x = state
	theta1, theta2 = x[1:2]
	l1, l2 = params.l1, params.l2
	maxlen = (l1+l2)*1.2
	x1,y1 = 0., 0.
	x2 = x1 - sin(theta1)*l1
	y2 = y1 - cos(theta1)*l1
	x3 = x2 - sin(theta1 + theta2)*l2
	y3 = y2 - cos(theta1 + theta2)*l2

	legend := false
	xlims := (-maxlen, maxlen)
	ylims := (-maxlen, maxlen)
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	# target line
	@series begin
		linecolor := :red
		linewidth := 5

		[0., 0.], [0., maxlen]
	end

	# arms
	@series begin
		linecolor := :black
		linewidth := 10

		[x1, x2, x3], [y1, y2, y3]
	end

end
