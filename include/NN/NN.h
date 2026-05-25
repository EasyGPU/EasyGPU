#pragma once

/**
 * @file NN.h
 * @brief Umbrella header for EasyGPU neural network training utilities.
 *
 * Include this single header for all NN functionality:
 * @code
 * #include <NN/NN.h>
 * using namespace GPU::NN;
 * @endcode
 */

#ifndef EASYGPU_NN_H
#define EASYGPU_NN_H

#include <NN/Attention.h>
#include <NN/Checkpoint.h>
#include <NN/Embedding.h>
#include <NN/Layers.h>
#include <NN/Loss.h>
#include <NN/Normalization.h>
#include <NN/Optimizer.h>
#include <NN/Tensor.h>
#include <NN/Transformer.h>

#endif // EASYGPU_NN_H
