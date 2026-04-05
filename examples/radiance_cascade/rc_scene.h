#pragma once

#include <vector>
#include <Utility/Vec.h>

namespace RC {

constexpr int IMAGE_WIDTH = 512;
constexpr int IMAGE_HEIGHT = 512;
constexpr int PROBE_SPACING = 16;
constexpr float BASE_INTERVAL_LENGTH = 0.05f;
constexpr int MAX_CASCADES = 8;
constexpr int CASCADE_SIZE = 512;



bool SaveImage(const std::vector<GPU::Math::Vec4>& pixels, const char* filename);

} // namespace RC
