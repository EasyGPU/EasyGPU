#pragma once

/**
 * @file ADCore.h
 * @brief Umbrella header for EasyGPU Automatic Differentiation.
 *
 * Include this single header to use all AD functionality:
 * @code
 * #include <AD/ADCore.h>
 * using namespace GPU::AD;
 * @endcode
 */

#ifndef EASYGPU_AD_CORE_H
#define EASYGPU_AD_CORE_H

#include <AD/ADKernel.h>
#include <AD/AdjointGenerator.h>
#include <AD/AdjointInspector.h>
#include <AD/AdjointKernel.h>
#include <AD/AdjointTable.h>
#include <AD/GradientTape.h>
#include <AD/TapeEntry.h>

#endif // EASYGPU_AD_CORE_H
