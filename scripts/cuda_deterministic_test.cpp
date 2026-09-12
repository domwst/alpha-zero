// Test-only preload: strict tensor parity needs deterministic GPU reductions.
// This library is never loaded by the production trainer.
#include <ATen/Context.h>

__attribute__((constructor)) static void deterministic_test() {
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setDeterministicAlgorithms(true, false);
}
