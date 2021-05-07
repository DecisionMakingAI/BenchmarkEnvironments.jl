function hiv_sim!(x, ϵ1, ϵ2, params, dt)
    T1, T2, T1i, T2i, V, E = x
    toti = T1i + T2i
    λ1, λ2, λE, d1, d2, dE, k1, k2, Kb, Kd, δ, δE, ρ1, ρ2, m1, m2, bE, NT, c, f = params

    dT1 = λ1 - (d1 * T1) - (1-ϵ1) * k1 * V * T1
    dT2 = λ2 - (d2 * T2) - (1-f*ϵ1) * k2 * V * T2
    dT1i = (1-ϵ1) * k1 * V * T1 - (δ * T1i) - m1 * E * T1i
    dT2i = (1-f*ϵ1) * k2 * V * T2 - (δ * T2i) - m2 * E * T2i
    dV = (1 - ϵ2) * NT * δ * toti - (c * V) - ((1-ϵ1) * ρ1 * k1 * T1 + (1 - f*ϵ1) * ρ2 * k2 * T2) * V
    dE = λE + bE * toti * E / (toti + Kb) - dE * toti * E / (toti + Kd) - δE * E
    x[1] += dt * dT1
    x[2] += dt * dT2
    x[3] += dt * dT1i
    x[4] += dt * dT2i
    x[5] += dt * dV
    x[6] += dt * dE
    return nothing
end

function hiv_simulate!(x, ϵ, params, dt, sim_steps)
    Δt = dt / sim_steps
    for i in 1:sim_steps
        hiv_sim!(x, ϵ[1], ϵ[2], params, Δt)
    end
    return nothing
end

function hiv_params()
    λ1 = 10000.0
    λ2 = 31.98
    λE = 1.0
    d1 = 0.01
    d2 = 0.01
    dE = 0.25
    k1 = 8.0 * 1e-7
    k2 = 1.0 * 1e-4
    Kb = 100.0
    Kd = 500.0
    δ = 0.7
    δE = 0.1
    ρ1 = 1.0
    ρ2 = 1.0
    m1 = 1.0 * 1e-5
    m2 = 1.0 * 1e-5
    bE = 0.3
    NT = 100.0
    c = 13.0
    f = 0.34
    return λ1, λ2, λE, d1, d2, dE, k1, k2, Kb, Kd, δ, δE, ρ1, ρ2, m1, m2, bE, NT, c, f
end

function hiv_dt(x, p, t)
    T1, T2, T1i, T2i, V, E = x
    toti = T1i + T2i
    ϵ1, ϵ2 = p
    dT1 = 10000.0 - (0.01 * T1) - (1.0 - ϵ1) * (8.0 * 1e-7) * V * T1
    dT2 = 31.98 - (0.01 * T2) - (1.0 - 0.34*ϵ1) * (1.0 * 1e-4) * V * T2
    dT1i = (1-ϵ1) * (8.0 * 1e-7) * V * T1 - (0.7 * T1i) - (1.0 * 1e-5) * E * T1i
    dT2i = (1-0.34*ϵ1) * (1.0 * 1e-4) * V * T2 - (0.7 * T2i) - (1.0 * 1e-5) * E * T2i
    dV = (1 - ϵ2) * 100.0 * 0.7 * (T1i + T2i) - (13.0 * V) - ((1-ϵ1) * 1.0 * (8.0 * 1e-7) * T1 + (1 - 0.34*ϵ1) * 1.0 * (1.0 * 1e-4) * T2) * V
    dE = 1.0 + 0.3 * (T1i + T2i) * E / (T1i + T2i + 100.0) - 0.25 * (T1i + T2i) * E / (T1i + T2i + 500.0) - 0.1 * E
    return @SVector [dT1, dT2, dT1i, dT2i, dV, dE]
end

function hiv_simulate!(prob, x, ϵ, dt)
    tspan = (0.0, dt)  # simulate 5 days at a time    
    prob = remake(prob, u0=x, p=ϵ)
    sol = DiffEqBase.solve(prob, BS3(), reltol=1e-1, abstol=1e-1, save_everystep=false)
    return sol.u[2]
end

function hiv_step!(prob::ODEProblem, state, action, dt, actions)
    ϵ = actions[action]
    t, s, x = state
    reward = hiv_reward(s, ϵ)
    s = hiv_simulate!(prob, s, ϵ, dt)
    t += dt
    @. x = log10(s)
    γ = 0.98
    if t ≥ 1000
        γ = 0.0
    end

    return (t, s, x), x, reward, γ

end

function hiv_step!(state, action, dt, actions, params)
    ϵ = actions[action]
    t, s, x = state
    reward = hiv_reward(s, ϵ)
    sim_steps = 5*65
    hiv_simulate!(s, ϵ, params, dt, sim_steps)
    # hiv_simulate!(s, ϵ, dt)
    t += dt
    @. x = log10(s)
    γ = 0.98
    if t ≥ 1000
        γ = 0.0
    end

    return (t, s, x), x, reward, γ

end

function hiv_reward(s, ϵ)
    V, E = s[5], s[6]
    ϵ1, ϵ2 = ϵ[1], ϵ[2]
    return 0.1 * V  + 20000.0 * ϵ1^2 + 2000.0 * ϵ2^2 - 1000.0 * E
end

function hiv_initial()
    t = 0.0
    # s = @SVector [163573.0, 5.0, 11945.0, 46.0, 63919.0, 24.0]
    s = [163573.0, 5.0, 11945.0, 46.0, 63919.0, 24.0]
    x = zeros(6)
    @. x = log10(s)
    return (t, s, x), x
end

function hiv()
    return create_hiv()
end

function create_hiv()
    [(3.0, 8.0), (-1, 4.0), (-2.5, 8.0), (-2.0, 4.0), (-1.0, 8.0), (0.0, 7.0)]
    X = Array{Float64,2}([
         3.0 8
        -1.0 4
        -2.5 8
        -2.0 4
        -1.0 8
         0.0 7
    ])
    S = ([0.0, 1000.0],
        exp10.(X),
        X
    )
    A = 1:4
    # u0 = @SVector [163573.0, 5.0, 11945.0, 46.0, 63919.0, 24.0]
    dt = 5.0
    # tspan = (0.0, dt)
    # ps = (0.0, 0.0)
    # prob = ODEProblem(hiv_dt,u0,tspan,ps)
    params = hiv_params()
    actions = ((0.0, 0.0), (0.7, 0.0), (0.0, 0.3), (0.7, 0.3))
    # p = (s,a)->hiv_step!(prob, s, a, dt, actions)
    p = (s,a)->hiv_step!(s, a, dt, actions, params)
    d0 = hiv_initial

    meta = Dict{Symbol,Any}()
    # meta[:minreward] = -5.0
    # meta[:maxreward] = 5
    # meta[:minreturn] = -5 * ceil(maxT / (dt * 20))  
    # meta[:maxreturn] =  5 * ceil(maxT / (dt * 20))  
    meta[:stochastic] = false
    meta[:minhorizon] = 200
    meta[:maxhorizon] = 200
    meta[:discounted] = true
    meta[:episodes] = 300
    
	bplot = HivPlotData()
    render = (state,clearplot=false)->hivplot(bplot,state,clearplot)
    m = SequentialProblem(S,X,A,p,d0,meta,render)
end

struct HivPlotData <: Any
    ts::Vector{Float64}
    xs::Vector{Vector{Float64}}	

	function HivPlotData() 
		new(Vector{Float64}(),Vector{Vector{Float64}}())
	end
end

@userplot HivPlot
@recipe function f(ap::HivPlot)
	data, state, clearplot = ap.args
	t, state, x = state
	if clearplot
        empty!(data.ts)
        empty!(data.xs)
	end

    push!(data.ts, t)
	push!(data.xs, deepcopy(x))
	legend := false
	xlims := (0, 1000)
	# grid := true
	# ticks := true
	layout := (3,2)
	# foreground_color := :white
	# aspect_ratio := 1.
    xs = hcat(data.xs...)
    names = ["T1 Healthy", "T2 Healthy", "T1 Infected", "T2 Infected", "Free Virus Particles", "Immune Effectors"]
    labels = ["log10 T1", "log10 T2", "log10 T1i", "log10 T2i", "log10 V", "log10 E"]
    ylimits = [(3.0, 8.0), (-1, 4.0), (-2.5, 8.0), (-2.0, 4.0), (-1.0, 8.0), (0.0, 7.0)]
    for i in 1:6
        @series begin 
            title --> string(names[i])
            ylims := ylimits[i]
            yguide := labels[i]
            seriestype := :path
            subplot := i
            data.ts, xs[i, :]
        end
    end

end