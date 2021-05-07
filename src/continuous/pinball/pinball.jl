# this code was closely ported from the rlpy implementaiton https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Pinball.py
# configs/*.cfg come directly from rlpy, but have been renamed making medium easy, simple_single is medium, and hard is hard.

struct PinBallConfig{T}
    start_pos::Tuple{T,T}
    target_pos::Tuple{T,T}
    target_radius::T
    ball_radius::T
    noise::T
    drag::T
    force::T

    function PinBallConfig(::Type{T}, start_pos::Tuple, target_pos::Tuple, target_radius) where {T}
        new{T}(convert.(T, start_pos), convert.(T, target_pos), T(target_radius), T(0.02), T(0.), T(0.995), T(1. / 5.))
    end
    function PinBallConfig(::Type{T}, start_pos::Tuple, target_pos::Tuple, target_radius, ball_radius) where {T}
        new{T}(convert.(T, start_pos), convert.(T, target_pos), T(target_radius), T(ball_radius), T(0.), T(0.995), T(1. / 5.))
    end
    function PinBallConfig(::Type{T}, start_pos::Tuple, target_pos::Tuple, target_radius, ball_radius, noise, drag, force) where {T}
        new{T}(convert.(T, start_pos), convert.(T, target_pos), T(target_radius), T(ball_radius), T(noise), T(drag), T(force))
    end
end

struct PinballObstacle{T} <: Any where {T}
    points::Array{Tuple{T,T}, 1}
    minx::T
    miny::T
    maxx::T
    maxy::T

    function PinballObstacle(::Type{T}, points::Array{Tuple{TX, TY}, 1}) where {T,TX,TY}
        pts = [T.(p) for p in points]
        minx, miny = min.(pts...)
        maxx, maxy = max.(pts...)
        new{T}(pts, minx, miny, maxx, maxy)
    end
end

function pinball_empty()
    return pinball_finitetime("pinball_empty.cfg", maxT=1000, stochastic_start=true, randomize=true, num_episodes=100)
end

function pinball_box()
    return pinball_finitetime("pinball_box.cfg", maxT=1000, stochastic_start=true, randomize=true, num_episodes=100)
end

function pinball_easy()
    return pinball_finitetime("pinball_easy.cfg", maxT=1000, stochastic_start=true, randomize=true, num_episodes=200)
end

function pinball_medium()
    return pinball_finitetime("pinball_medium.cfg", maxT=2000, stochastic_start=true, randomize=true, num_episodes=400, threshold=8500)
end

function pinball_hard()
    return pinball_finitetime("pinball_hard.cfg", maxT=5000, stochastic_start=true, randomize=true, num_episodes=1000, threshold=8000)
end


function pinball_finitetime(config::String; maxT=1000, stochastic_start=false, randomize=false, num_episodes=100, threshold=9000)
    X = zeros((4,2))
    X[1,:] .= [0., 1.]  # x range
	X[2,:] .= [0., 1.]  # y range
	X[3,:] .= [-2., 2.] # xdot range
    X[4,:] .= [-2., 2.] # ydot range
    S = ([0.0, maxT],
        X
    )
    A = 1:5

    obstacles, conf = read_config(Float64, config)
    if randomize
        conf = pinball_randomize(conf)
    end
    x = zeros(4)
    x .= conf.start_pos[1], conf.start_pos[2], 0.0, 0.0
    # ball = BallState(Float64, (conf.start_pos[1], conf.start_pos[2]), conf.ball_radius)
    dt = 0.05
    # x = zeros(4)

    p = (s,a)->pinball_step!(s, a, conf, obstacles, dt, stochastic_start, maxT)
    d0 = ()->pinball_d0!(conf.start_pos, stochastic_start)

    meta = Dict{Symbol,Any}()
    meta[:minreward] = -5.0
    meta[:maxreward] = 10000.0
    meta[:minreturn] = -5 * ceil(maxT / (dt * 20))  # time moves at 20*dt per step
    meta[:maxreturn] = 10000  # actually lower than this, but if you started in the goal state this would be the case. 
    meta[:stochastic] = true
    meta[:minhorizon] = 40  # not sure the the true minimum is. This seems like a good lower bound
    meta[:maxhorizon] = ceil(maxT / (dt * 20))
    meta[:discounted] = false
    meta[:episodes] = num_episodes
    meta[:threshold] = threshold
    render = (state,clearplot=false)->pinballplot(state, obstacles, conf)
    m = SequentialProblem(S,X,A,p,d0,meta,render)
    
	return m    
end


function pinball_randomize(conf::PinBallConfig)
	target_radius = conf.target_radius + rand(Uniform(-0.1, 0.1))  * conf.target_radius  # scaled target radius randomly by standard deviation of 10% of the specified radius
    ball_radius = conf.ball_radius + rand(Uniform(-0.1, 0.1)) * conf.ball_radius
    noise = rand() * 0.25    # chance for random action
    drag = 1. - exp(rand(Uniform(log(0.001), log(0.1))))  # friction coefficient for ball
    force =  rand(Uniform(0.1, 0.3))  # force applied to the ball
	conf = PinBallConfig(Float64, conf.start_pos, conf.target_pos, target_radius, ball_radius, noise, drag, force)
    return conf
end

function pinball_step!(state, action, config, obstacles, dt, stochastic_start, maxT)
    # pinball_update!($state, $action, $config, $obstacles, $dt, $maxT)
    t,x = state
    t, x, reward, γ = pinball_update!(t,x, action, config, obstacles, dt, maxT)
    if γ == 0.0
        reset_ball!(x, config.start_pos, stochastic_start)
        t = 0.0
    end

    return (t,x), x, reward, γ
    # return nothing
end

function pinball_d0!(start_pos, stochastic_start)
    x = zeros(4)
    reset_ball!(x, start_pos, stochastic_start)
    # reset_ball!(ball, start_pos, stochastic_start)
    # update_observation!(x, ball)
    t = 0.0
    return (t,x), x
end

function reset_ball!(x, start_pos::Tuple, stochastic_start::Bool)
    x[1] = start_pos[1]
    x[2] = start_pos[2]
    # ball.x = start_pos[1]
	# ball.y = start_pos[2]
	if stochastic_start
		# ball.x += 0.02 * randn()
        # ball.y += 0.02 * randn()
        x[1] += 0.02 * randn()
		x[2] += 0.02 * randn()
	end
	# ball.xDot = 0.
    # ball.yDot = 0.
    x[3] = 0.0
    x[4] = 0.0
    return nothing
end

function pinball_sim_update!(ball, dt, ball_radius, obstacles, iteration::Int)
    stepball!(ball, dt, ball_radius)

    ncollision = 0
    dxdy = (0.0, 0.0)

    for obs in obstacles
        hit, double_collision, intercept = collision(obs, ball, ball_radius)
        if hit
            dxdy = dxdy .+ collision_effect(ball, hit, double_collision, intercept)
            ncollision += 1
        end
    end
    if ncollision == 1
        # ball.xDot = dxdy[1]
        # ball.yDot = dxdy[2]
        ball[3:4] .= dxdy
        if iteration == 19
            stepball!(ball, dt, ball_radius)
        end
    elseif ncollision > 1
        ball[3] = -ball[3]
        ball[4] = -ball[4] 
        # ball.xDot = -ball.xDot
        # ball.yDot = -ball.yDot
    end
    return nothing
end


function pinball_update!(t, ball, action::Int, config::PinBallConfig{T}, obstacles::Array{PinballObstacle{T},1}, dt::T, maxT) where {T}
    if action <= 0 || action > 5
        error("Action needs to be an integer in [1, 5]")
    end

	if rand() < config.noise
		action = rand(1:5)
	end
    # add action effect
    if action == 1
        add_impulse!(ball, config.force, 0.)  # Acc x
    elseif action ==2
        add_impulse!(ball, 0., -config.force) # Dec y
    elseif action ==3
        add_impulse!(ball, -config.force, 0.) # Dec x
    elseif action == 4
        add_impulse!(ball, 0., config.force) # Acc y
    else
        add_impulse!(ball, 0., 0.)  # No action
    end

    reward = 0.0
    for i in 1:20
        pinball_sim_update!(ball, dt, config.ball_radius, obstacles, i)
		t += dt
        found_goal = at_goal(ball, config)
		done = found_goal || t > maxT


        if done
            reward = (10000. * found_goal) - (1 - found_goal)

            return t,ball,reward,0.0
        end
    end

    add_drag!(ball, config.drag)
    checkbounds!(ball)

    if action == 5
        reward = -1.
    else
        reward = -5.
    end

    return t,ball,reward,1.0
end

function checkbounds!(ball)
    x = ball[1]
    y = ball[2]
    if x > 1.0
        ball[1] = 0.95
    elseif x < 0.0
        ball[1] = 0.05
    end
    if y > 1.0
        ball[2] = 0.95
    elseif y < 0.0
        ball[2] = 0.05
    end
    return nothing
end

function at_goal(ball, config::PinBallConfig)
    res = √sum(@. ((ball[1], ball[2]) - config.target_pos)^2)
    return res < config.target_radius
end

function add_impulse!(ball, Δx::T, Δy::T) where {T}
    xDot = ball[3] + Δx
    yDot = ball[4] + Δy
    ball[3] = clamp(xDot, -2., 2.)
    ball[4] = clamp(yDot, -2., 2.)
    return nothing
end

function add_drag!(ball, drag::T) where {T}
    ball[3] *= drag
    ball[4] *= drag
    return nothing
end


function stepball!(ball, dt::T, radius) where {T}
    ball[1] += ball[3] * (radius * dt)
    ball[2] += ball[4] * (radius * dt)
    return nothing
end


function read_config(::Type{T}, source) where {T}
	source = joinpath(@__DIR__, "configs", source)
    obstacles = Array{PinballObstacle{T}, 1}()
    target_pos = Tuple{T,T}((0.,0.))
    target_radius = T(0.04)
    ball_radius = T(0.02)
    start_pos = Tuple{T,T}((0.,0.))
    noise = T(0.)
    drag = T(0.995)
    force = T(1. / 5.)
    lines = readlines(source)
    for line in lines
        tokens = split(strip(line))
        if length(tokens) <= 0
            continue
        elseif tokens[1] == "polygon"
			nums = map(x->parse(T, x), tokens[2:end])
			points = [(x,y) for (x,y) in zip(nums[1:2:end], nums[2:2:end])]
            push!(obstacles, PinballObstacle(T, points))
        elseif tokens[1] == "target"
            target_pos = Tuple{T,T}(map(x->parse(T,x), tokens[2:3]))
            target_radius = parse(T,tokens[4])
        elseif tokens[1] == "start"
            start_pos = Tuple{T,T}(map(x->parse(T,x), tokens[2:3]))
        elseif tokens[1] == "ball"
            ball_radius = parse(T, tokens[2])
        end
    end

    conf = PinBallConfig(T, start_pos, target_pos, target_radius, ball_radius, noise, drag, force)

    return obstacles, conf
end

function collision(obs::PinballObstacle{T}, ball, radius) where {T}
    double_collision = false
    intercept_found = false
    intercept = ((0.0,0.0),(0.0,0.0))
    if ball[1] - radius > obs.maxx
        return intercept_found, double_collision, intercept
	end
	if ball[1] + radius < obs.minx
        return intercept_found, double_collision, intercept
	end
    if ball[2] - radius > obs.maxy
        return intercept_found, double_collision, intercept
	end
    if ball[2] + radius < obs.miny
        return intercept_found, double_collision, intercept
	end
    

    i = 1
    j = 2
    while i ≤ length(obs.points)
        p1, p2 = obs.points[i], obs.points[j]
        if intercept_edge(p1, p2, ball, radius)
            if intercept_found
                intercept = select_edge((p1, p2), intercept, ball)
                double_collision = true
            else
                intercept = (p1, p2)
                intercept_found = true
            end
        end
        i += 1
        j += 1
        if j > length(obs.points)
            j = 1
        end
    end
    return intercept_found, double_collision, intercept
end

function collision_effect(ball, intercept_found::Bool, double_collision::Bool, intercept::Tuple{Tuple{T,T},Tuple{T,T}})::Tuple{T,T} where {T}
    if double_collision
        return -ball[3], -ball[4]
    end

    obstacle_vector = intercept[2] .- intercept[1]
    if obstacle_vector[1] < 0.
        obstacle_vector = intercept[1] .- intercept[2]
    end

    velocity_vector = (ball[3], ball[4])
    θ = compute_angle(velocity_vector, obstacle_vector) - π
    if θ < 0.
        θ += 2π
    end

    intercept_theta = compute_angle((-1, 0), obstacle_vector)
    θ += intercept_theta

    velocity = √sum(velocity_vector.^2)
    # velocity = norm(velocity_vector)

    return velocity * cos(θ), velocity * sin(θ)
end

function compute_angle(v1, v2)
    angle_diff = atan(v1[1], v1[2]) - atan(v2[1], v2[2])
    if angle_diff < 0.
        angle_diff += 2π
    end
    return angle_diff
end

function intercept_edge(p1::Tuple{T,T}, p2::Tuple{T,T}, ball, radius) where {T}
    edge = p2 .- p1
    pball = (ball[1], ball[2])
    difference = pball .- p1

    scalar_proj = dot(difference, edge) / dot(edge, edge)
    scalar_proj = clamp(scalar_proj, 0., 1.)

    closest_pt = p1 .+ (edge .* scalar_proj)
    obstacle_to_ball = pball .- closest_pt
    distance = dot(obstacle_to_ball, obstacle_to_ball)

    if distance <= radius^2
        # collision if the ball is not moving away
        velocity = (ball[3], ball[4])
        ball_to_obstacle = closest_pt .- pball

        angle = compute_angle(ball_to_obstacle, velocity)
        if angle > π
            angle = 2π - angle
        end

        if angle > (π / 1.99)
            return false
        end
        return true
    else
        return false
    end
end

function select_edge(intersect1::Tuple{Tuple{T,T},Tuple{T,T}}, intersect2::Tuple{Tuple{T,T},Tuple{T,T}}, ball) where {T}
    velocity = (ball[3], ball[4])
    obstacle_vector1 = intersect1[2] .- intersect1[1]
    obstacle_vector2 = intersect2[2] .- intersect2[1]
    angle1 = compute_angle(velocity, obstacle_vector1)
    if angle1 > π
        angle1 -= π
    end

    angle2 = compute_angle(velocity, obstacle_vector2)
    if angle2 > π
        angle2 -= π
    end

    if abs(angle1 - π / 2.) < abs(angle2 - π / 2.)
        return intersect1
    else
        return intersect2
    end
end

@userplot PinBallPlot
@recipe function f(ap::PinBallPlot)
    state, obstacles, config = ap.args
    t, ball = state
    ballx = ball[1]
    bally = ball[2]
    bradius = config.ball_radius
	tx, ty = config.target_pos
	tr = config.target_radius


	legend := false
	xlims := (0., 1.)
	ylims := (0., 1.)
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	# obstacles
	for ob in obstacles
		@series begin
			seriestype := :shape
			seriescolor := :blue
			xpts = [p[1] for p in ob.points]
			ypts = [p[2] for p in ob.points]

			xpts, ypts
		end
	end

	# goal
	@series begin
		seriestype := :shape
		linecolor := nothing
		seriescolor := :green
		aspect_ratio := 1.
		fillalpha := 0.4

		circle_shape(tx, ty, tr)
	end

	# ball
	@series begin
		seriestype := :shape
		linecolor := nothing
		seriescolor := :red
		aspect_ratio := 1.

		circle_shape(ballx, bally, bradius)
	end

end
