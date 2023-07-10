# BenchmarkEnvironments

[![Build Status](https://github.com/DecisionMakingAI/BenchmarkEnvironments.jl/workflows/CI/badge.svg)](https://github.com/DecisionMakingAI/BenchmarkEnvironments.jl/actions)
[![Coverage](https://codecov.io/gh/DecisionMakingAI/BenchmarkEnvironments.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DecisionMakingAI/BenchmarkEnvironments.jl)


This repository contains environments useful for experimenting and benchmarking reinforcement learning algorithms. This library is under active development and is subject to change. 

The list of environments and their constructors (with defaults) are below:
```julia
using BenchmarkEnvironments


# Discrete State & Action Environments
num_states = 10
env = simple_chain(num_states, stochastic=false, failchance=0.1, randomize=false, hard=false)
env = simple_chain_finitetime(num_states, stochastic=false, failchance=0.1, randomize=false, droptime=true, hard=false, reward=:negative, discount=1.0)

# Continuous State Environments
env = acrobot_finitetime(randomize=false, maxT=400.0, dt=0.2, Atype=:Discrete, droptime=true, stochastic_start=false)
env = cartpole_finitetime(randomize=false, tMax=20.0, dt=0.02, Atype=:Discrete, droptime=true)
env = mountaincar_finitetime(randomize=false, maxT=5000, Atype=:Discrete, droptime=true, stochastic_start=false)
env = hiv() # HIV simulator
env = pinball_empty() # basically a continuous state gridworld with acceleration
env = pinball_box()  # 1 large obstical in the middle of the room
env = pinball_easy()  # a realitively simple pinball environment
env = pinball_medium() # pinball with moderate difficult of obsticals
env = pinball_hard()  # a fairly challenging environment that is difficult to solve with random exploration
env = ballandbeam_fixedgoal(randomize=false, maxT=20.0, dt=0.05, droptime=true, stochastic_start=true, action_type=:Discrete)  # ball and beam environment with fixed goal location
env = ballandbeam_randomgoal(randomize=false, maxT=20.0, dt=0.05, droptime=true, stochastic_start=true, action_type=:Discrete)  # ball and beam environment with random goal position
env = ballandbeam_tacking(randomize=false, maxT=20.0, dt=0.05, droptime=true, stochastic_start=true, action_type=:Discrete)  # ball and beam environment where the goal moves during the episode
env = pendulum_finitetime(randomize=false, maxT=20.0, dt=0.01, droptime=true, stochastic_start=true)  # single arm pendulum environment
```