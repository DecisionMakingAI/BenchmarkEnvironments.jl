
# function : current state s, current action a, state space S maps to next state s2
function simplechain_transition(s,a,S)
    if a == 1  # go left
        s2 = s-1 # decrease state value by 1
    elseif a == 2 # go right
        s2 = s+1 # increase state value by 1
    elseif a == -1
        s2 = s
    else
        error("action must be {1,2}, but was : ", a)
    end
    if s2 > maximum(S)  # if s2 is larger than the maximum state
        s2 = minimum(S) # set it to the start state (smallest state)
    elseif s2 < minimum(S) # if s2 is smaller than smallest state
        s2 = minimum(S) # make it the smallest state
    end
    return s2  # return next state
end

# function : current state s, current action a, state space S maps to next state s2
function hardchain_transition(s,a,S)
    if a == 1  # go left
        s2 = minimum(S) # decrease state value by 1
    elseif a == 2 # go right
        s2 = s+1 # increase state value by 1
    elseif a == -1
        s2 = s
    else
        error("action must be {1,2}, but was : ", a)
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

function simplechain_step(p, r, s, a, S)
    s′ = p(s,a,S)
    maxS = maximum(S)
    minS = minimum(S)
    γ = 1.0
    if s==maxS && s′==minS
        γ = 0.0
    end
    reward = r(s,a,s′)
    
    return s′,s′,reward,γ
end

function create_simple_chain(num_states::Int; stochastic=false, failchance=0.1, hard=false)
    S = 1:num_states
    A = 1:2

    if hard
        tfun = hardchain_transition
    else
        tfun = simplechain_transition
    end 
    r = (s,a,s′)-> -1.0
    if !stochastic
        p = (s,a)->simplechain_step(tfun, r, s, a, S)  # mask out S in above function    
    else
        p = (s,a)->simplechain_step(tfun, r, s, simplechain_perturb_action(a,failchance), S)
    end

    d0 = ()->(1,1)
    X = S
    meta = Dict{Symbol,Any}()
    meta[:minreward] = -1.0
    meta[:maxreward] = -1.0
    meta[:minreturn] = -Inf
    meta[:maxreturn] = -Float64(num_states)
    meta[:stochastic] = stochastic
    meta[:minhorizon] = num_states
    meta[:maxhorizon] = Inf
    meta[:discounted] = false
    render = (state,clearplot=false)->nothing
    m = SequentialProblem(S,X,A,p,d0,meta,render)
    return m
end

function simple_chain(num_states::Int; stochastic=false, failchance=0.1, randomize=false, hard=false)
    if randomize
        failchance = rand() * 0.25
    end
    return create_simple_chain(num_states, stochastic=stochastic, failchance=failchance, hard=hard)
end

function finite_horizon_transition(m0, s, a, maxT)
    t = s[1]
    t += 1
    snum = s[2] 
    s′ = m0.p(snum,a)
    if t > maxT || m0.γ(snum, a, s′)==0.0
        t = 1
        s′ = m0.d0()
    end

    return (t, s′)
end

function finitehorizon_chainstep(p, r, s, a, maxT, S, discount)
    t,x = s
    t += 1 
    x′ = p(x,a)
    maxS = maximum(S)
    minS = minimum(S)
    γ = 1.0 * discount
    if t > maxT || (x==maxS && x′==minS)
        γ = 0.0
    end
    reward = r(x,a,x′)
    
    return t,x′,reward,γ
end

function finitehorizon_chainstep2(p, rmode, s, a, maxT, S, discount)
    t,x = s
    t += 1 
    x′ = p(x,a)
    maxS = maximum(S)
    minS = minimum(S)
    γ = 1.0 * discount
    if t > maxT || (x==maxS && x′==minS)
        γ = 0.0
    end
    if rmode == :negative
        reward = -1.0
    elseif rmode == :sparse
        reward = chain_sparsereward(x, a, x′, maxS)
    else
        throw("Unrecognized reward function: $rmode")
    end
    
    return t,x′,reward,γ
end

function chain_constreward(s,a,s′)
    return -1.0
end

function chain_scaledreward(s,a,s′,k)
    return -1.0 * s/k
end

function chain_sparsereward(s,a,s′,numS)
    if s == numS && s′ == 1
        return 1.0
    else
        return 0.0
    end
end

function create_simple_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, droptime=true, hard=false, reward=:negative, discount=1.0)
    maxT = num_states * 20
    S = (1:maxT, 1:num_states)
    A = 1:2
    meta = Dict{Symbol,Any}()

    if reward == :negative    
        meta[:minreward] = -1.0
        meta[:maxreward] = -1.0
    elseif reward == :sparse
        meta[:minreward] = 0.0
        meta[:maxreward] = 100.0
    else
        throw("Unexpected reward mode: $reward")
    end
    if reward ∈ [:negative, :sparse]
        nothing
    else
        throw("Unexpected reward mode: $reward")
    end
    
    
    if hard
        tfun = (s,a)->hardchain_transition(s,a,S[2])
    else
        tfun = (s,a)->simplechain_transition(s,a,S[2])
    end 
    

    if !stochastic
        pfun = (s,a)->finitehorizon_chainstep2(tfun, reward, s, a, maxT, S[2], discount) 
    else
        pfun = (s,a)->finitehorizon_chainstep2(tfun, reward, s, simplechain_perturb_action(a,failchance), maxT, S[2], discount) 
    end

    if droptime
		X = S[2]
        function get_outcome1(pfun, s,a)
            t, x, r, γ = pfun(s,a)
			s = (t,x)
            return s,x,r,γ
		end
        p = (s,a)->get_outcome1(pfun, s, a)
        d0 = ()->((1,1),1)
	else
		X = S
		function get_outcome2(pfun,s,a)
			t, x, r, γ = pfun(s,a)
			s = (t,x)
			return s,s,r,γ
		end
        p = (s,a)->get_outcome2(pfun, s, a)
        d0 = ()->((1,1),(1,1))
	end

    
    meta[:minreward] = -1.0
    meta[:maxreward] = -1.0
    meta[:minreturn] = -Float64(maxT)
    meta[:maxreturn] = -Float64(num_states)
    meta[:stochastic] = stochastic
    meta[:minhorizon] = num_states
    meta[:maxhorizon] = maxT
    meta[:discounted] = discount != 1.0
    
    render = (state,clearplot=false)->nothing

	m = SequentialProblem(S,X,A,p,d0,meta,render)

    return m
end

function simple_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, randomize=false, droptime=true, hard=false, reward=:negative, discount=1.0)
    if randomize
        failchance = rand() * 0.25
    end
    return create_simple_chain_finitetime(num_states, stochastic=stochastic, failchance=failchance, droptime=droptime, hard=hard, reward=reward, discount=discount)
end