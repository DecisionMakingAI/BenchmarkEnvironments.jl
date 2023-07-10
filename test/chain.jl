using BenchmarkEnvironments
using Test
using Distributions
using DecisionMakingEnvironments


@testset "Simple Chain Tests" begin
    prob = simple_chain(3; stochastic=false, failchance=0.1, randomize=false)
    @test typeof(prob) <: SequentialProblem
    @test length(prob.A) == 2
    @test eltype(prob.A) <: Int
    @test length(prob.S) == 3
    @test eltype(prob.S) <: Int
    @test prob.d0() isa Tuple{Int,Int}
    @test prob.d0() == (1,1)
    @test prob.p(prob.d0()[1],2)[1] isa Int
    @test prob.p(prob.d0()[1],2)[2] isa Int
    @test prob.p(prob.d0()[1],2)[3] isa Float64
    @test prob.p(prob.d0()[1],2)[4] isa Float64
    for s in prob.S
        for a in prob.A
            @test prob.p(s,a) == sample(prob, s, a)
        end
    end
    @test prob.p(1,2)[1] == 2
    @test prob.p(3,2)[1] == 1
    @test prob.p(1,2)[3] == -1.0
    @test prob.p(1,2)[4] == 1.0
    @test prob.p(3,2)[4] == 0.0
    @test prob.p(1,1)[1] == 1

    meta = meta_information(prob)
    @test meta[:minreturn] isa Float64
    @test meta[:minreward] == -1
    @test meta[:maxreturn] == -3
    @test meta[:minhorizon] == 3
    @test meta[:maxhorizon] == Inf
    prob = simple_chain(3, stochastic=true)
    @test prob.p(1,2)[1] isa Int
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
