

struct PendulumParams{T} <:Any where {T<:Real}
    m::T 	# mass link
    l::T 	# length link
	mu::T    # friction constant
	g::T   	# gravity force (not directional)
	# fmag::T # max force magnitude

    PendulumParams() = new{Float64}(1., 1., 0.1, 9.8)
    PendulumParams(T::Type) = new{T}(1., 1., 0.1, 9.8)
	PendulumParams(T::Type, m, l, mu, g) = new{T}(m, l, mu, g)
end



function random_pendulum_params()
	m = 1.0
	l = 1.0
	g = 9.8
	mu = 0.1
	m = rand(Uniform(0.9*m, 1.1*m))  # mass of arm
	l = rand(Uniform(0.9*l, 1.1*l))    # length of arm
	mu = rand(Uniform(0.9*mu, 1.1*mu))  # friction coeff
	g = rand(Uniform(8., 10.))  # gravity
	params = PendulumParams(Float64, m, l, mu, g)
	return params
end

function pendulum_finitetime(;randomize=false, maxT=20.0, dt=0.01, droptime=true, stochastic_start=true)
	if randomize
		params = PendulumParams()
	else
		params = random_pendulum_params()
	end
	return create_pendulum_finitetime(params, maxT=maxT, dt=dt, droptime=droptime, stochastic_start=stochastic_start)
end

function create_pendulum_finitetime(params; maxT=20.0, dt=0.01, droptime=true, stochastic_start=true)
	X = zeros((2,2))
	X[1,:] .= [-π, π]           # theta1 range
	X[2,:] .= [-3. * π, 3. * π] # theta1Dot range
	S = ([0.0, maxT], X)

	A = [-1.0, 1.0]
	sim_steps = 10
	if droptime
		function get_outcome1(s,a,params,dt,sim_steps,maxT,stochastic_start)
			t, x, r, γ = pendulum_step!(s,a, params, dt, sim_steps, maxT, stochastic_start)
			s = (t,x)
			return s,x,r,γ
		end
        p = (s,a)->get_outcome1(s,a,params,dt,sim_steps,maxT,stochastic_start)
        
        function pdd0_obs()
            t,x = pendulum_sample_initial(stochastic_start)
            return (t,x), x
        end
        d0 = pdd0_obs
	else
		X = S
		function get_outcome2(s,a,params,dt,sim_steps, maxT,stochastic_start)
			t, x, r, γ = pendulum_step!(s,a, params, dt, sim_steps, maxT, stochastic_start)
			s = (t,x)
			return s,s,r,γ
		end
        p = (s,a)->get_outcome2(s,a,params,dt,sim_steps,maxT,stochastic_start)
        function pdd0()
            s = pendlum_sample_initial(stochastic_start)
            return s,s
        end
		d0 = pdd0
    end
    
    
	meta = Dict{Symbol,Any}()
    meta[:minreward] = -2.0
    meta[:maxreward] = 2.0
    meta[:minreturn] = -2 * maxT
    meta[:maxreturn] = 2 * maxT
    meta[:stochastic] = false
    meta[:minhorizon] = ceil(Int, maxT / dt)
    meta[:maxhorizon] = ceil(Int, maxT / dt)
	meta[:discounted] = false
	
	render = state->pendulumplot(state, params)

	m = SequentialProblem(S,X,A,p,d0,meta,render)
end

function pendulum_sim!(state, u, params::PendulumParams, dt, simSteps::Int)
	theta, omega = state
	m, l, mu, g = params.m, params.l, params.mu, params.g

	thetaDot = 0.
	omegaDot = 0.
	subDt = dt / simSteps

	for i in 1:simSteps
		thetaDot = omega
		omegaDot = (-mu * omega - m*g*l*sin(theta) + u) / (m*l^2)
		theta += subDt*thetaDot
		omega += subDt*omegaDot

	end

	theta = mod(theta + π, 2. * π) - π
	omega = clamp(omega, -3. * π, 3. * π)

	state .= theta, omega
	return nothing
end


function pendulum_step!(state, action, params::PendulumParams, dt, simSteps, maxT, stochastic_start)
	u = clamp(action, -1., 1.) * 10.0
	t,x = state
	pendulum_sim!(x, u, params, dt, simSteps)
	t += dt

	reward = -cos(x[1]) - (x[2]^2 / 100.)
	γ = 1.0
	done = t ≥ maxT

	if done
        γ = 0.0
        t, x = pendulum_sample_initial(stochastic_start)
	end
	
	return t, x, reward, γ
end



function pendulum_sample_initial(stochastic_start=true)
	state = zeros(2)
	if stochastic_start
		state[1] = rand(Uniform(-π, π))
	end
    
	return 0.0, state
end

@userplot PendulumPlot
@recipe function f(ap::PendulumPlot)
    state, params = ap.args
    t,x = state
    theta = x[1]
	l = params.l
	maxlen = (l)*1.2
	x1,y1 = 0., 0.
	x2 = x1 - sin(theta)*l
	y2 = y1 - cos(theta)*l


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

		[x1, x2], [y1, y2]
	end

end