module BenchmarkEnvironments

using DecisionMakingEnvironments
using RecipesBase, StaticArrays, Distributions, LinearAlgebra
using DiffEqBase, OrdinaryDiffEq

include("utils.jl")

# Discrete environments
include("discrete/discrete.jl")
export simple_chain, simple_chain_finitetime, hard_chain_finitetime

#continuous state environments
include("continuous/continuous.jl")
export acrobot_finitetime, cartpole_finitetime, mountaincar_finitetime
export pinball_finitetime, pinball_box, pinball_empty, pinball_easy, pinball_medium, pinball_hard
export bicycle_balance, bicycle_goal
export hiv
export pendulum_finitetime
export ballandbeam_fixedgoal, ballandbeam_randomgoal, ballandbeam_tacking
end
