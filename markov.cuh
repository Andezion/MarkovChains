#ifndef MARKOV_CUH
#define MARKOV_CUH

#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(call) { \
const cudaError_t error = call; \
if (error != cudaSuccess) { \
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
exit(1); \
} \
}

class MarkovChainCUDA {
    int num_states;
    int num_agents;

    float* d_transition_matrix{};
    int* d_current_states{};
    void* d_rng_states{};

public:
    MarkovChainCUDA(int states, int agents);
    ~MarkovChainCUDA();

    void setTransitionMatrix(const std::vector<float>& matrix_host);

    void setStates(const std::vector<int>& states_host);

    void getStates(std::vector<int>& states_host);

    void initRNG(unsigned long seed) const;

    void step() const;
};

#endif // MARKOV_CUH