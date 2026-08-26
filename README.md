# Physics informed adaptive quantum reservor
1. `pi_qrc.py` <- simple 4-qubit RC learning (in physics-informed way) to do FFT
2. `adaptive_pi_qrc` <-added noise context (Lindbladian) and auxiliary NN(noise), learning to optimize measurements selection & RC parameters
3. `benchmark_adaptive_selector.py` <- benchmark comparing adaptive RC with a static measurement policy
