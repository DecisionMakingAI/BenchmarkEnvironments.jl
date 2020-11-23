# Bicycle domain ported from rlpy https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Bicycle.py
# original paper Learning to Drive a Bicycle using Reinforcement Learning and Shaping, Jette Randlov, Preben Alstrom, 1998. https://www.semanticscholar.org/paper/Learning-to-Drive-a-Bicycle-Using-Reinforcement-and-Randl%C3%B8v-Alstr%C3%B8m/10bad197f1c1115005a56973b8326e5f7fc1031c

# Bicycle domain might not be correct.
struct BicycleParams{T} <:Any where {T<:Real}
	g::T    	# gravity 9.82
	v::T		# velocity of the bicycle (10 km/h paper value) default is 10. / 3.6 (10kph appox 6.2mph and 10kph is 2.77778 meters per second or 10/3)
	d_CM::T 	# vertical distance between the CM for the bicycle and for the cyclist 0.3m
	c::T    	# horizontal distance between the point where the front wheel touches the ground and the CM 0.66 m
	h::T		# height of the CM over the ground  0.94 cm
	M_c::T  	# mass bicycle  9kg ≈ 19.8lbs 15kg ≈ 33lbs (default is 15kg)
	M_d::T		# mass tire 1.0kg ≈ 2.2lbs 1.5kg ≈ 3.3lbs 3.0kg ≈ 4.4lbs (default is 1.7)
	M_p::T		# mass cyclist M_p = 45.4kg ≈ 100lbs 60.0kg ≈ 132.0lbs 90.7kg ≈ 200lbs (default is 60kg)
	M::T   		# mass of cyclist and bike M_p + M_c
	r::T	   	# radius of tire 0.34m
	dsigma::T  	# angular velcity of a tire (v/r)
	I::T    	# moment of inertia for bicycle and cyclist 13/3 * M_c * h^2 + M_p * (h + d_CM)^2
	I_dc::T 	# moment of inertia for tire M_d * r^2
	I_dv::T 	# moment of inertia for tire 3/2 * M_d * r^2
	I_dl::T 	# moment of inertia for tire 1/2 * M_d * r^2
	l::T    	# disance between the front and tire and the back tire at the point where they both touch the ground 1.11m

    function BicycleParams(T::Type=Float64)
		g = T(9.82)
		v = T(10. / 3.6)
		d_CM = T(0.3)
		c = T(0.66)
		h = T(0.94)
		M_c = T(15.)
		M_d = T(1.7)
		M_p = T(60.)
		M = M_c + M_p
		r = T(0.34)
		dsigma = T(v / r)
		I = T(13. / 3. * M_c * h^2 + M_p * (h+d_CM)^2)
		I_dc = T(M_d * r^2)
		I_dv = T(3. / 2.)
		I_dl = T(M_d / 2. * r^2)
		l = T(1.11)
		new{T}(g, v, d_CM, c, h, M_c, M_d, M_p, M, r, dsigma, I, I_dc, I_dv, I_dl, l)
	end

	function BicycleParams(::Type{T}, g, M_c, M_d, M_p, v) where {T}
		d_CM = T(0.3)
		c = T(0.66)
		h = T(0.94)
		M = M_c + M_p
		r = T(0.34)
		dsigma = T(v / r)
		I = T(13. / 3. * M_c * h^2 + M_p * (h+d_CM)^2)
		I_dc = T(M_d * r^2)
		I_dv = T(3. / 2. * M_d * r^2)
		I_dl = T(M_d / 2. * r^2)
		l = T(1.11)
		new{T}(g, v, d_CM, c, h, M_c, M_d, M_p, M, r, dsigma, I, I_dc, I_dv, I_dl, l)
	end
end



function random_bike_params()
	g = rand(Uniform(8., 10.))  		# gravity
	v = rand(Uniform(8., 15.)) / 3.6	# velocity of the bicycle mps
	M_c = rand(Uniform(9., 15.))  		# mass bicycle  9kg ≈ 19.8lbs 15kg ≈ 33lbs (default is 15kg)
	M_d = rand(Uniform(1., 3.))	    	# mass tire 1.0kg ≈ 2.2lbs 1.5kg ≈ 3.3lbs 3.0kg ≈ 4.4lbs (default is 1.7)
	M_p = rand(Uniform(45.4, 90.7))		# mass cyclist M_p = 45.4kg ≈ 100lbs 60.0kg ≈ 132.0lbs 90.7kg ≈ 200lbs (default is 60kg)

	params = BicycleParams(Float64, g, M_c, M_d, M_p, v)
	return params
end

function bicycle_balance(;randomize=false, maxT=30.0)
	if randomize
		params = random_bike_params()
	else
		params = BicycleParams()
	end
	m = create_bicycle_balance(params, maxT=maxT, dt=0.01)
end

function bicycle_goal(;randomize=false, maxT=30.0)
	if randomize
		params = random_bike_params()
	else
		params = BicycleParams()
	end
	m = create_bicycle_goal(params, maxT=maxT, dt=0.01)
end

function bicycle_spaces(maxT)
	X1 = zeros((5,2))
    X1[1,:] .= [-π * 12. / 180., π * 12 / 180]
	X1[2,:] .= [-π, π]
	X1[3,:] .= [-π * 80. / 180., π * 80. / 180.]
	X1[4,:] .= [-π, π]
	X1[5,:] .= [-π, π]
	X2 = zeros(2,2)
	X2[1,:] .= [-Inf, Inf]
	X2[2,:] .= [-Inf, Inf]

	S = ([0.0, maxT], X1, X2)
	A = 1:9
	return S, A
end

function create_bicycle_balance(params; maxT=60.0, dt=0.01)
	S, A = bicycle_spaces(maxT)
	X = S[1]

	p = (s,a)->bicycle_balance_step!(s, a, params, dt, maxT)
	d0 = bicycle_balance_initial
	
	meta = Dict{Symbol,Any}()
    # meta[:minreward] = -5.0
    # meta[:maxreward] = 5
    # meta[:minreturn] = -5 * ceil(maxT / (dt * 20))  
    # meta[:maxreturn] =  5 * ceil(maxT / (dt * 20))  
    meta[:stochastic] = true
    meta[:minhorizon] = 1  # not sure the the true minimum is. This seems like a good lower bound
    meta[:maxhorizon] = ceil(Int, maxT / dt)
    meta[:discounted] = false
	meta[:episodes] = 1000
	bplot = BicyclePlotData()
    render = (state,clearplot=false)->bicycleplot(bplot,state,clearplot)
    m = SequentialProblem(S,X,A,p,d0,meta,render)
	return m
end

function create_bicycle_goal(params; maxT=600.0, dt=0.01)
	S, A = bicycle_spaces(maxT)
	X = S[1]

	p = (s,a)->bicycle_goal_step!(s, a, params, dt, maxT)
	d0 = bicycle_goal_initial
	
	meta = Dict{Symbol,Any}()
    # meta[:minreward] = -5.0
    # meta[:maxreward] = 5
    # meta[:minreturn] = -5 * ceil(maxT / (dt * 20))  
    # meta[:maxreturn] =  5 * ceil(maxT / (dt * 20))  
    meta[:stochastic] = true
    meta[:minhorizon] = 1  # not sure the the true minimum is. This seems like a good lower bound
    meta[:maxhorizon] = ceil(Int, maxT / dt)
    meta[:discounted] = false
	meta[:episodes] = 3000
	bplot = BicyclePlotData()
    render = (state,clearplot=false)->bicycleplot(bplot,state,clearplot)
    m = SequentialProblem(S,X,A,p,d0,meta,render)
	return m
end

function bicycle_balance_initial()
	t = 0.0
	x = zeros(5)
	pos = zeros(2)
	
	return (t,x,pos), x
end

function bicycle_goal_initial()
	t = 0.0
	x = zeros(5)
	pos = zeros(2)
	pos[1] = -500.0  # 20 * randn()
	pos[2] = 0.0 # 20 * randn()
	return (t,x,pos), x
end


function bicycle_sim!(state, pos, constants::BicycleParams{T}, u::T, d::T, dt::T) where {T<:Real}
	omega, omegaDot, theta, thetaDot, psi = state
	g = constants.g
	v = constants.v
	d_CM = constants.d_CM
	c = constants.c
	h = constants.h
	M_c = constants.M_c
	M_d = constants.M_d
	M_p = constants.M_p
	M = constants.M
	r = constants.r
	dsigma = constants.dsigma
	I = constants.I
	I_dc = constants.I_dc
	I_dv = constants.I_dv
	I_dl = constants.I_dl
	l = constants.l

	w = 0.02 * (2.0 * rand() - 1.0)


	phi = omega + atan(d+w) / h
	invr_f = abs(sin(theta)) / l
	invr_b = abs(tan(theta)) / l
	if theta != 0.
		invr_CM = ((l-c)^2 + invr_b^(-2))^(-0.5)
	else
		invr_CM = 0.
	end

	nomega = omega + dt * omegaDot
	nomegaDot = omegaDot + dt * (M * h * g * sin(phi) - cos(phi) * (I_dc * dsigma * thetaDot + sign(theta) * v^2 * (M_d * r * (invr_f + invr_b) + M * h * invr_CM))) / I
	out = theta + dt * thetaDot
	
	if abs(out) < (π * 80. / 180.)
		ntheta = out
		nthetaDot = thetaDot + dt * (u - I_dv * dsigma * omegaDot) / I_dl
	else
		ntheta = sign(out) * (π * 80. / 180.)
		nthetaDot = 0.
	end
	# ntheta = mod(ntheta + π, 2. * π) - π

	npsi = psi + dt * sign(theta) * v * invr_b

	npsi = npsi % (2 * π)
	if npsi > π
		npsi -= 2 * π
	end
	

	# dposxf = v * dt * -sin(psi + theta + sign(psi + theta) * asin(v*dt*0.5*invr_f))
	# dposyf = v * dt * cos(psi + theta + sign(psi + theta) * asin(v*dt*0.5*invr_f))
	dposxb = v * dt * -sin(psi + sign(psi) * asin(v*dt*0.5*invr_b))
	dposyb = v * dt * cos(psi + sign(psi) * asin(v*dt*0.5*invr_b))

	state[1] = nomega
	state[2] = omegaDot = nomegaDot
	state[3] = ntheta
	state[4] = nthetaDot
	state[5] = npsi
	pos[1] += dposxb
	pos[2] += dposyb

	return npsi - psi

end

function compute_actions(action::Int)
	if action <= 0 || action > 10
        error("Action needs to be an integer in [1, 9]")
    end
    u = 0.
	d = 0.
	uaction = floor(Int, (action-1) / 3) + 1
	daction = action % 3

	if uaction == 1
        u = -2.
    elseif uaction == 2
    	u =  0.
	else
		u = 2.
    end

	if daction == 1
		d = -0.02
	elseif daction == 2
		d =  0.
	else
		d = 0.02
	end
	return u,d
end

function bicycle_balance_step!(state, action::Int, params, dt, maxT)
    u,d = compute_actions(action)
	t,x,pos = state
	psi_diff = bicycle_sim!(x, pos, params, u, d, dt)
	t += dt
	
	done = bike_fell(x) || t ≥ maxT
	
	reward = 1
	
	γ = 1.0
	if done
		γ = 0.0
	end

    return (t, x, pos), x, reward, γ
end

function bicycle_goal_step!(state, action::Int, params, dt, maxT)
    u,d = compute_actions(action)
	t,x,pos = state
	psi_diff = bicycle_sim!(x, pos, params, u, d, dt)
	t += dt
	
	if bike_fell(x)
		reward = -1.0
		γ = 0.0
	elseif bike_at_goal(pos)
		reward = 0.01
		γ = 0.0
	elseif t ≥ maxT
		reward = (4.0 - atan(pos...)^2) * 0.00004
		γ = 0.0
	else
		reward = (4.0 - atan(pos...)^2) * 0.00004
		γ = 1.0
	end
	
    return (t, x, pos), x, reward, γ
end

function bike_fell(x)
	omega = x[1]
	fallcond = abs(omega) > (π * 12. / 180.) # 12 degrees in rads
	return fallcond
end

function bike_at_goal(pos)
	d = √sum(pos.^2)
	return d ≤ 10  # within 10 meters
end


mutable struct BicyclePlotData <: Any
	ts::Vector{Float64}
	ωs::Vector{Float64}
	xs::Vector{Float64}
	ys::Vector{Float64}

	function BicyclePlotData() 
		new(Vector{Float64}(),Vector{Float64}(),Vector{Float64}(),Vector{Float64}())
	end
end

@userplot BicyclePlot
@recipe function f(ap::BicyclePlot)
	data, state, clearplot = ap.args
	t, x, pos = state
	if clearplot
		empty!(data.ts)
		empty!(data.ωs)
		empty!(data.xs)
		empty!(data.ys)
	end

	push!(data.ts, t)
	push!(data.ωs, x[1])
	push!(data.xs, pos[1])
	push!(data.ys, pos[2])
		

	legend := false
	# xlims := (0., 1.)
	# ylims := (0., 1.)
	grid := true
	ticks := true
	layout := (2,1)
	foreground_color := :white
	# aspect_ratio := 1.

	@series begin 
		seriestype := :path
		seriescolor := :blue
		subplot := 1
		# xlims := (0., 30.)
		fall = π * 12.0 / 180.0
		ylims := (-fall, fall)
		data.ts, data.ωs
	end

	@series begin
		seriestype := :path
		seriescolor := :black
		subplot := 2
		# xlims := (-5, 5)
		# ylims := (-3, 10)
		data.xs, data.ys
	end

	# goal
	@series begin
		seriestype := :shape
		linecolor := nothing
		seriescolor := :green
		aspect_ratio := 1.
		fillalpha := 0.4
		subplot := 2
		circle_shape(0.0, 0.0, 10)
	end

	# current pos
	@series begin
		seriestype := :shape
		linecolor := nothing
		seriescolor := :red
		aspect_ratio := 1.

		circle_shape(pos[1], pos[2], 1.0)
	end

end
