using BenchmarkEnvironments
using Test
using Distributions
using DecisionMakingEnvironments


@testset "Simple Chain Tests" begin
    prob = simple_chain(3; stochastic=false, failchance=0.1, randomize=false)
    @test typeof(prob) <: MDP
    @test length(prob.A) == 2
    @test eltype(prob.A) <: Int
    @test length(prob.S) == 3
    @test eltype(prob.S) <: Int
    @test prob.d0() isa Int
    @test prob.d0() == 1
    @test prob.p(prob.d0(),2) isa Int
    @test prob.p(1,2) == 2
    @test prob.p(3,2) == 1
    @test prob.r(1,2,1) isa Float64
    @test prob.r(1,2,1) == -1.0
    @test prob.γ(prob.d0(),2,1) isa Float64
    @test prob.γ(3,1,2) == 1.0
    @test prob.γ(3,2,1) == 0.0
    meta = meta_information(prob)
    @test meta[:minreturn] isa Float64
    @test meta[:minreward] == -1
    @test meta[:maxreturn] == -3
    @test meta[:minhorizon] == 3
    @test meta[:maxhorizon] == Inf
    prob = simple_chain(3, stochastic=true)
    @test prob.p(1,2) isa Int
end

@testset "Simple Chain Finitetime Tests" begin
    prob = simple_chain_finitetime(3, stochastic=false, droptime=false)
    @test typeof(prob) <: SequentialProblem
    maxT = 3*20
    @test length(prob.A) == 2
    @test eltype(prob.A) <: Int
    @test length(prob.S) == 2
    @test length(prob.S[1]) == maxT
    @test length(prob.S[2]) == 3
    @test eltype(prob.S[1]) <: Int
    @test eltype(prob.S[2]) <: Int
    s0, x = prob.d0()
    @test s0 isa Tuple{Int,Int}
    @test s0[1] isa Int
    @test s0[2] isa Int
    @test x == s0
    @test s0[1] == 1
    @test s0[2] == 1
    s1, x, r, γ = sample(prob, s0, 2)

    @test s1 isa Tuple{Int,Int}
    @test x == s1
    
    prob = simple_chain_finitetime(3, stochastic=false, droptime=true)
    s1, x, r, γ = sample(prob, s0, 2)
    @test s1 isa Tuple{Int,Int}
    @test x isa Int
    @test x == s1[1]

    
    meta = meta_information(prob)
    @test meta[:minreturn] isa Float64
    @test meta[:minreturn] == -maxT
    @test meta[:minreward] == -1
    @test meta[:maxreward] == -1
    @test meta[:maxreturn] == -3
    @test meta[:minhorizon] == 3
    @test meta[:maxhorizon] == maxT
end
