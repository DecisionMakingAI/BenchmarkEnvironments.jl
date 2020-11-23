
# function : current state s, current action a, state space S maps to next state s2
function simplechain_transition(s,a,S)
    if a == 1  # go left
        s2 = s-1 # decrease state value by 1
    elseif a == 2 # go right
        s2 = s+1 # increase state value by 1
    elseif a == -1
        s2 = s
    else
        error("action must be {1,2}, but was : ", action)
    end
    if s2 > maximum(S)  # if s2 is larger than the maximum state
        s2 = minimum(S) # set it to the start state (smallest state)
    elseif s2 < minimum(S) # if s2 is smaller than smallest state
        s2 = minimum(S) # make it the smallest state
    end
    return s2  # return next state
end

function simplechain_perturb_action(a::Int, failchance)
    ϵ = rand()
    if ϵ ≤ 1.0 - failchance
        return a
    elseif ϵ ≤ 1.0 - failchance / 2
        return Int(-1)
    else
        return a==1 ? 2 : 1
    end
end


function create_simple_chain(num_states::Int; stochastic=false, failchance=0.1)
    S = 1:num_states
    A = 1:2

    if !stochastic
        p = (s,a)->simplechain_transition(s,a,S)  # mask out S in above function    
    else
        p = (s,a)->simplechain_transition(s,simplechain_perturb_action(a,failchance), S)
    end
    d0 = ()->minimum(S)
    r = (s,a,s′)-> -1.0
    γ = (s,a,s′)-> (s==num_states && s′==1) ? 0.0 : 1.0
    meta = Dict{Symbol,Any}()
    meta[:minreward] = -1.0
    meta[:maxreward] = -1.0
    meta[:minreturn] = -Inf
    meta[:maxreturn] = -Float64(num_states)
    meta[:stochastic] = stochastic
    meta[:minhorizon] = num_states
    meta[:maxhorizon] = Inf
    meta[:discounted] = false
    m = MDP(S,A,p,r,γ,d0,meta)
    return m
end

function simple_chain(num_states::Int; stochastic=false, failchance=0.1, randomize=false)
    if randomize
        failchance = rand() * 0.25
    end
    return create_simple_chain(num_states, stochastic=stochastic, failchance=failchance)
end

function finite_horizon_transition(m0, s, a, maxT)
    t = s[1]
    t += 1
    s′ = m0.p(s[2],a)
    if t > maxT || m0.γ(s[2], a, s′)==0.0
        t = 1
        s′ = m0.d0()
    end

    return (t, s′)
end

function create_simple_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, droptime=true)
    maxT = num_states * 20
    S = (1:maxT, 1:num_states)
    A = 1:2
    m0 = create_simple_chain(num_states; stochastic=stochastic, failchance=failchance)
    p = (s,a)->finite_horizon_transition(m0, s, a, maxT)
    γ = (s,a,s′)-> (s[2]==num_states && s′[2]==1) || (s[1] ≥ s′[1]) ? 0.0 : 1.0
    r = (s,a,s′)->m0.r(s[2],a,s′[2])
    d0 = ()->(1,m0.d0())
    meta = Dict{Symbol,Any}()
    meta[:minreward] = -1.0
    meta[:maxreward] = -1.0
    meta[:minreturn] = -Float64(maxT)
    meta[:maxreturn] = -Float64(num_states)
    meta[:stochastic] = stochastic
    meta[:minhorizon] = num_states
    meta[:maxhorizon] = maxT
    meta[:discounted] = false
    
    if droptime
        X = S[2]
        obs = s->s[2]
        m = POMDP(S,A,X,p,obs,r,γ,d0,meta)
    else
        m = MDP(S,A,p,r,γ,d0,meta)
    end

    return m
end

function simple_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, randomize=false, droptime=true)
    if randomize
        failchance = rand() * 0.25
    end
    return create_simple_chain_finitetime(num_states, stochastic=stochastic, failchance=failchance, droptime=droptime)
end


