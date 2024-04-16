
struct MountainCarParams{T} <:Any where {T<:Real}
    ucoeff::T # action coefficient for acceleration
    g::T # gravity coeff
    h::T # cosine frequency parameter

    MountainCarParams() = new{Float64}(0.001, 0.0025, 3.)
    MountainCarParams(T::Type) = new{T}(T(0.001), T(0.0025), T(3.))
	MountainCarParams(T::Type, ucoeff, g, h) = new{T}(T(ucoeff), T(g), T(h))
end



function random_mountaincar_params()
	ubase = 0.001
	u = rand(rng, Uniform(0.8*ubase, 1.2*ubase))  # coefficient for action force
	# fix these to keep the problem around the same difficulty
	g = 0.0025   	# "gravity" parameter
	h = 3.  		# mountain frequency parameter  (fixed because it changes the state width. Need to recompute xlims)

	params = MountainCarParams(T, u, g, h)
	return params
end

function mountaincar_finitetime(;randomize=false, maxT=5000, Atype=:Discrete, droptime=true, stochastic_start=false)
	if randomize
		params = random_mountaincar_params()
	else
		params = MountainCarParams()
	end
	return create_finitetime_mountaincar(params; maxT=maxT, Atype=Atype, droptime=droptime, stochastic_start=stochastic_start)
end

function create_finitetime_mountaincar(params::MountainCarParams; maxT=5000, Atype=:Discrete, droptime=true, stochastic_start=false)
	X = zeros((2,2))
    X[1,:] .= [-1.2, 0.5]       # x range
	X[2,:] .= [-0.07, 0.07]     # xDot range
    S = ([0. maxT],				# time range
		X)           	
						
	if Atype==:Discrete
		A = 1:3
	else
		A = [-1.0 1.0]
    end
    dt = 1.0
	if droptime
		X = S[2]
		function get_outcome1(s,a,params,dt,maxT,stochastic_start)
			t, x, r, γ = mountaincar_step!(s,a, params, dt, maxT, stochastic_start)
			s = (t,x)
			return s,x,r,γ
		end
		p = (s,a)->get_outcome1(s,a,params,dt,maxT,stochastic_start)
		function mcd0_obs()
			t,x = mc_sample_initial(stochastic_start)
			return (t,x), x
		end
		d0 = mcd0_obs
	else
		X = S
		function get_outcome2(s,a,params,dt,maxT,stochastic_start)
			t, x, r, γ = mountaincar_step!(s,a, params, dt, maxT,stochastic_start)
			s = (t,x)
			return s,s,r,γ
		end
		p = (s,a)->get_outcome2(s,a,params,dt,maxT,stochastic_start)
		function mcd0()
			s = mc_sample_initial(stochastic_start)
			return s,s
		end
		d0 = mcd0
    end
    
	meta = Dict{Symbol,Any}()
    meta[:minreward] = -1.0
    meta[:maxreward] = -1.0
    meta[:minreturn] = -ceil(maxT)
    meta[:maxreturn] = -80
    meta[:stochastic] = false
    meta[:minhorizon] = 80
    meta[:maxhorizon] = ceil(Int, maxT)
	meta[:discounted] = false
	meta[:episodes] = 100
	meta[:threshold] = -150  # maybe this should depend on environment params
	
	render = (state,clearplot=false)->mountaincarplot(state, params)
	
	m = SequentialProblem(S,X,A,p,d0,meta, render)
	return m
end


function mountaincar_sim!(state, u, params)
	x, xDot = state
	xDot = xDot + params.ucoeff * u - params.g * cos(params.h * x)
	xDot = clamp(xDot, -0.07, 0.07)
	x += xDot

	if x < -1.2
		x = -1.2
		xDot = 0.
	end

	state[1:2] .= x, xDot
end

function mcget_torque(action::Int)
    if action <= 0 || action > 3
        error("Action needs to be an integer in [1, 3]")
    end
    u = 0.0
    u = (Float64(action) - 2.0)
    return u
end

function mcget_torque(action::Float64)
    u = clamp(action, -1., 1.)
    return u
end

function mc_sample_initial(stochastic_start)
	t = 0.0
	
	x = zeros(2)
	x[1] = -0.5
	x[2] = 0.
	
	if stochastic_start
		x[1] = rand(Uniform(-0.6, -0.4))
	end
    
    return t, x
end

function mountaincar_step!(state, action, params, dt, maxT, stochastic_start)
    u = mcget_torque(action)
    t, x = state
	mountaincar_sim!(x, u, params)
	t += 1

    reward = -1.0
    γ = 1.0
	done = mc_terminal(t, x, maxT)
	if done
        γ = 0.0
        t, x = mc_sample_initial(stochastic_start)
	end

	return t, x, reward, γ
end


function mc_terminal(t, state, maxT)::Bool
	goalcond = state[1] ≥ 0.5
	timecond = t ≥ maxT
	done = goalcond | timecond

	return done
end

@userplot MountainCarPlot
@recipe function f(ap::MountainCarPlot)
	state, params = ap.args
	t, (x, xvel) = state
	h = params.h
	y = sin(h * x)

	xpts = range(-1.2, stop=0.5, length=50)
	ypts = map(x->sin(h*x), xpts)

	legend := false
	xlims := (-1.2, 0.5)
	ylims := (min(ypts...)-0.05, max(ypts...)+0.05)
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	# mountain
	@series begin
		seriestype := :line
		linecolor := :black
		linewidth := 2

		xpts, ypts
	end

	# car
	@series begin
		seriestype := :shape
		linecolor := nothing
		seriescolor := :blue
		aspect_ratio := 1.

		circle_shape(x, y, 0.05)
	end
end
