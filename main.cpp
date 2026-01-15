#include "markov.cuh"

int main()
{
    MarkovChainCUDA mc(5, 10000);

    mc.setTransitionMatrix(transition_probs);

    mc.initRNG(seed);

    mc.step();
    return 0;
}