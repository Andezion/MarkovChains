#include "markov.cuh"
#include <curand_kernel.h>
#include <iostream>

__global__ void init_rng_kernel(curandState* state, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void markov_step_kernel(
    int* current_states,
    const float* transition_matrix,
    curandState* rng_states,
    int num_states,
    int num_agents
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_agents) return;


    int current_state = current_states[id];


    curandState localState = rng_states[id];
    float random_val = curand_uniform(&localState);
    rng_states[id] = localState;

    const float* probs = &transition_matrix[current_state * num_states];

    float cumulative = 0.0f;
    int next_state = num_states - 1;

    for (int i = 0; i < num_states; ++i) {
        cumulative += probs[i];
        if (random_val <= cumulative) {
            next_state = i;
            break;
        }
    }

    current_states[id] = next_state;
}

MarkovChainCUDA::MarkovChainCUDA(int states, int agents)
    : num_states(states), num_agents(agents) {

    CHECK_CUDA(cudaMalloc(&d_transition_matrix, states * states * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_current_states, agents * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rng_states, agents * sizeof(curandState)));
}

MarkovChainCUDA::~MarkovChainCUDA() {
    cudaFree(d_transition_matrix);
    cudaFree(d_current_states);
    cudaFree(d_rng_states);
}

void MarkovChainCUDA::setTransitionMatrix(const std::vector<float>& matrix_host) {
    CHECK_CUDA(cudaMemcpy(d_transition_matrix, matrix_host.data(),
               num_states * num_states * sizeof(float), cudaMemcpyHostToDevice));
}

void MarkovChainCUDA::setStates(const std::vector<int>& states_host) {
    CHECK_CUDA(cudaMemcpy(d_current_states, states_host.data(),
               num_agents * sizeof(int), cudaMemcpyHostToDevice));
}

void MarkovChainCUDA::getStates(std::vector<int>& states_host) {
    CHECK_CUDA(cudaMemcpy(states_host.data(), d_current_states,
               num_agents * sizeof(int), cudaMemcpyDeviceToHost));
}

void MarkovChainCUDA::initRNG(unsigned long seed) {
    int blockSize = 256;
    int gridSize = (num_agents + blockSize - 1) / blockSize;
    init_rng_kernel<<<gridSize, blockSize>>>((curandState*)d_rng_states, seed, num_agents);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void MarkovChainCUDA::step() {
    int blockSize = 256;
    int gridSize = (num_agents + blockSize - 1) / blockSize;

    markov_step_kernel<<<gridSize, blockSize>>>(
        d_current_states,
        d_transition_matrix,
        (curandState*)d_rng_states,
        num_states,
        num_agents
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}