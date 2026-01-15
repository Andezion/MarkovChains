#include "markov.cuh"
#include <cassert>
#include <iostream>
#include <vector>

void test_deterministic_transition() {
    std::cout << "Running test_deterministic_transition... ";

    std::vector<float> matrix = {
        0.0f, 1.0f,
        1.0f, 0.0f
    };

    MarkovChainCUDA mc(2, 10);
    mc.setTransitionMatrix(matrix);
    mc.initRNG(42);

    std::vector<int> start_states(10, 0);
    mc.setStates(start_states);

    mc.step();
    std::vector<int> res;
    res.resize(10);
    mc.getStates(res);

    for(int s : res) assert(s == 1);

    mc.step();
    mc.getStates(res);
    for(int s : res) assert(s == 0);

    std::cout << "PASSED" << std::endl;
}

void test_stationary_distribution() {
    std::cout << "Running test_stationary_distribution... ";

    int n_agents = 10000;
    MarkovChainCUDA mc(2, n_agents);

    std::vector<float> matrix = {0.5f, 0.5f, 0.5f, 0.5f};
    mc.setTransitionMatrix(matrix);
    mc.initRNG(999);

    std::vector<int> states(n_agents, 0);
    mc.setStates(states);

    for(int i=0; i<50; ++i) mc.step();

    mc.getStates(states);

    int count0 = 0;
    for(int s : states) if(s == 0) count0++;

    float ratio = (float)count0 / n_agents;

    assert(ratio > 0.48f && ratio < 0.52f);

    std::cout << "PASSED (Ratio state 0: " << ratio << ")" << std::endl;
}

int main()
{
    test_deterministic_transition();
    test_stationary_distribution();
    return 0;
}