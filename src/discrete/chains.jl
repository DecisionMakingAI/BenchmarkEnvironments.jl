
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
        s2 = mimimum(S) # decrease state value by 1
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


function create_simple_chain(num_states::Int; stochastic=false, failchance=0.1, hard_transition=false)
    S = 1:num_states
    A = 1:2

    if hard_transition
        tfun = hard_transition
    else
        tfun = simplechain_transition
    end 

    if !stochastic
        p = (s,a)->tfun(s,a,S)  # mask out S in above function    
    else
        p = (s,a)->tfun(s,simplechain_perturb_action(a,failchance), S)
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
    m = MDP(S,A,p,r,γ,d0,meta, ()->nothing)
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
    snum = s[2] 
    s′ = m0.p(snum,a)
    if t > maxT || m0.γ(snum, a, s′)==0.0
        t = 1
        s′ = m0.d0()
    end

    return (t, s′)
end

function finitehorizon_chainstep(p, r, s, a, maxT, S)
    t,x = s
    t += 1 
    x′ = p(x,a)
    maxS = maximum(S)
    minS = minimum(S)
    γ = 1.0
    if t > maxT || (x==maxS && x′==minS)
        γ = 0.0
    end
    reward = r(x,a,x′)
    
    return t,x′,reward,γ
end

function chain_constreward(s,a,s′)
    return -1.0
end

function chain_scaledreward(s,a,s′,k)
    return -1.0 * s/k
end

function create_simple_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, droptime=true, scale_reward=false)
    maxT = num_states * 20
    S = (1:maxT, 1:num_states)
    A = 1:2
    meta = Dict{Symbol,Any}()

    
    rfun = (s,a,s′)->-1.0#chain_scaledreward(s,a,s′,1.0)#Float64(-1.0)
    meta[:minreward] = -1.0
    meta[:maxreward] = -1.0
    # if scale_reward
    #     # rfun = (s,a,s′)-> Float64(-1.0) / (Float64(s[2]) / Float64(num_states))
    #     rfun = (s,a,s′)->chain_scaledreward(s,a,s′,Float64(num_states))
    #     meta[:minreward] = -1.0
    #     meta[:maxreward] = -1.0 / num_states
    # end

    tfun = (s,a)->simplechain_transition(s,a,S[2])

    if !stochastic
        pfun = (s,a)->finitehorizon_chainstep(tfun, rfun, s, a, maxT, S[2]) 
    else
        pfun = (s,a)->finitehorizon_chainstep(tfun, rfun, s, simplechain_perturb_action(a,failchance), maxT, S[2]) 
    end

    if droptime
		X = S[2]
        # function get_outcome1(pfun,s,a)
        function get_outcome1(s,a,maxT,S)
            # @code_warntype (pfun(s,a))
            t, x, r, γ = finitehorizon_chainstep(tfun, rfun, s, a, maxT, S) 
            # t, x, r, γ = pfun(s,a)
			s = (t,x)
            return s,x,r,γ
            # return s,s[2],-1.0,1.0
		end
        p = (s,a)->get_outcome1(s, a, maxT, S[2])
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
    meta[:discounted] = false
    
    render = (state,clearplot=false)->nothing

	m = SequentialProblem(S,X,A,p,d0,meta,render)

    return m
end

function simple_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, randomize=false, droptime=true, scale_reward=false)
    if randomize
        failchance = rand() * 0.25
    end
    return create_simple_chain_finitetime(num_states, stochastic=stochastic, failchance=failchance, droptime=droptime, scale_reward=scale_reward)
end


function create_hard_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, droptime=true, scale_reward=false)
    maxT = num_states * 20
    S = (1:maxT, 1:num_states)
    A = 1:2
    m0 = create_simple_chain(num_states; stochastic=stochastic, failchance=failchance, hardchain_transition)
    p = (s,a)->finite_horizon_transition(m0, s, a, maxT)
    γ = (s,a,s′)-> (s[2]==num_states && s′[2]==1 && a==2) || (s[1] ≥ s′[1]) ? 0.0 : 1.0
    
    meta = Dict{Symbol,Any}()

    if scale_reward
        r = (s,a,s′)->m0.r(s[2],a,s′[2]) / (s[2] / num_states)
        meta[:minreward] = -1.0
        meta[:maxreward] = -1.0 / num_states
    else
        r = (s,a,s′)->m0.r(s[2],a,s′[2])
        meta[:minreward] = -1.0
        meta[:maxreward] = -1.0
    end
    d0 = ()->(1,m0.d0())
    
    meta[:minreturn] = -Float64(maxT)
    meta[:maxreturn] = -Float64(num_states)
    meta[:stochastic] = stochastic
    meta[:minhorizon] = num_states
    meta[:maxhorizon] = maxT
    meta[:discounted] = false
    
    if droptime
        X = S[2]
        obs = s->s[2]
        m = POMDP(S,A,X,p,obs,r,γ,d0,meta, ()->nothing)
    else
        m = MDP(S,A,p,r,γ,d0,meta, ()->nothing)
    end

    return m
end

function hard_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, randomize=false, droptime=true, scale_reward=false)
    if randomize
        failchance = rand() * 0.25
    end
    return create_hard_chain_finitetime(num_states, stochastic=stochastic, failchance=failchance, droptime=droptime, scale_reward=scale_reward)
end
