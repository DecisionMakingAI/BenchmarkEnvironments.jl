struct CartPoleParams{T} <:Any where {T<:Real}
    m::T # mass of pole
    l::T # length of pole
    mc::T # mass of cart
    muc::T # some constant?
    mup::T # some constant?
    fmag::T # magnitude of force applied
	g::T  # gravity (not signed)
    CartPoleParams() = new{Float64}(0.1, 0.5, 1., 0.0005, 0.000002, 10., 9.8)
    CartPoleParams(T::Type) = new{T}(T(0.1), T(0.5), T(1.), T(0.0005), T(0.000002), T(10.), T(9.8))
	CartPoleParams(T::Type, m, l, mc, g) = new{T}(m, l, mc, T(0.0005), T(0.000002), T(10.), g)
end

function cartpole_sim(state, constants::CartPoleParams, u, dt)
	x, xDot, theta, thetaDot = view(state, 1:4)
	omegaDot = 0.
	vDot = 0.
	m = constants.m     # mass of pole
    l = constants.l     # length of pole
    mc = constants.mc   # mass of cart
    muc = constants.muc # some constant?
    mup = constants.mup # some constant?
	g = constants.g
	# for i in 1:steps
	omegaDot = (g * sin(theta) + cos(theta) * (muc * sign(xDot) - u - m * l * thetaDot^2 * sin(theta)) / (m + mc) - mup * thetaDot / (m * l)) / (l * (4.0 / 3.0 - m / (m + mc) * cos(theta)^2))
    vDot = (u + m * l * (thetaDot^2 * sin(theta) - omegaDot*cos(theta)) - muc*sign(xDot)) / (m + mc)
    theta += dt * thetaDot
    thetaDot += dt * omegaDot
    x += dt * xDot
    xDot += dt * vDot

    theta = mod(theta + π, 2 * π) - π

	return x, xDot, theta, thetaDot
end

function cartpole_compute_torque(p::CartPoleParams, action::Int)
	if action <= 0 || action > 2
		error("Action needs to be an integer in [1, 2]")
	end
	u = 0.
	if action == 1
		u = -p.fmag
	else
		u =  p.fmag
	end
	return u
end

function cartpole_compute_torque(p::CartPoleParams, action)
	return clamp(action, -1.0, 1.0) * p.fmag
end


function cartpole_step(state, action, params, dt, maxT)
	t,x = state
	u = cartpole_compute_torque(params, action)
	x .= cartpole_sim(x, params, u, dt)

	t += dt
	γ = 1.0
	if cartpole_terminal(x) || (t > maxT - 1e-8)
		fill!(x, 0.0)
		t = 0.0
		γ = 0.0
	end
	r = 1.0
	return t,x,r, γ
end

function cartpole_finitetime(;randomize=false, tMax=20.0, dt=0.02, Atype=:Discrete, droptime=true)
	if randomize
		params = random_cartpole_params()
	else
		params = CartPoleParams()
	end
	return create_finitetime_cartpole(params; tMax=tMax, dt=dt, Atype=Atype, droptime=droptime)
end

function create_finitetime_cartpole(params::CartPoleParams; tMax=20.0, dt=0.02, Atype=:Discrete, droptime=true)
	S = ([0. tMax],				# time range
		[	-2.4 2.4 ;       	# x range
			-10. 10. ;       	# xDot range
			-π/12.0 π/12.0 ; 	# theta range
			-π π ])           	# thetaDot range
						
	if Atype==:Discrete
		A = 1:2
	else
		A = [-1.0 1.0]
	end
	

	if droptime
		X = S[2]
		function get_outcome1(s,a,params,dt,tMax)
			t, x, r, γ = cartpole_step(s,a, params, dt, tMax)
			s = (t,x)
			return s,x,r,γ
		end
		p = (s,a)->get_outcome1(s,a,params,dt,tMax)
		function sample_initialdrop()
			x = zeros(4)
			return (0.0, x), x
		end
		d0 = sample_initialdrop
	else
		X = S
		function get_outcome2(s,a,params,dt,tMax)
			t, x, r, γ = cartpole_step(s,a, params, dt, tMax)
			s = (t,x)
			return s,s,r,γ
		end
		p = (s,a)->get_outcome2(s,a,params,dt,tMax)
		function sample_initial()
			x = zeros(4)
			return (0.0, x), (0.0, x)
		end
		d0 = sample_initial
	end
	
	meta = Dict{Symbol,Any}()
    meta[:minreward] = 1.0
    meta[:maxreward] = 1.0
    meta[:minreturn] = 9.0
    meta[:maxreturn] = ceil(tMax / dt)
    meta[:stochastic] = false
    meta[:minhorizon] = 9.0
    meta[:maxhorizon] = ceil(tMax / dt)
	meta[:discounted] = false
	meta[:episodes] = 100
	
	render = (state,clearplot=false)->cartpoleplot(state, params)

	m = SequentialProblem(S,X,A,p,d0,meta,render)
	return m
end

function random_cartpole_params(::Type{T}=Float64) where {T}
	m = rand(Uniform(T(0.025), T(.25)))  # mass of pole
	l = rand(Uniform(T(0.1), T(1.0)))    # length of pole
	mc = rand(Uniform(T(0.1), T(5.)))    # mass of cart
	g = rand(Uniform(T(8.), T(10.)))     # gravity
	params = CartPoleParams(T, m, l, mc, g)
	return params
end

function cartpole_terminal(state)
	polecond = abs(state[3]) > (π / 15.0)
	cartcond = abs(state[1]) ≥ 2.4
	done = polecond | cartcond 
	return done
end

# CartPole Rendering originally from JuliaML/Reinforce.jl and has only be modified slighlty
# https://github.com/JuliaML/Reinforce.jl/blob/master/src/envs/cartpole.jl
@userplot CartPolePlot
@recipe function f(cp::CartPolePlot)
	state, params = cp.args
	t, (x, xvel, θ, θvel) = state
	legend := false
	xlims := (-2.4, 2.4)
	l = params.l
	ylims := (-Inf, 2l)
	grid := false
	ticks := nothing

	# pole
	@series begin
		linecolor := :red
		linewidth := 10
		[x, x + 2l * sin(θ)], [0.0, 2l * cos(θ)]
	end

	# cart
	@series begin
		seriescolor := :black
		seriestype := :shape
		hw = 0.5
		l, r = x-hw, x+hw
		t, b = 0.0, -0.1
		[l, r, r, l], [t, t, b, b]
	end
end