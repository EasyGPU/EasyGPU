#include "rc_scene.h"
#include <Utility/Vec.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>

namespace RC {

bool SaveImage(const std::vector<GPU::Math::Vec4>& pixels, const char* filename) {
    std::vector<unsigned char> img(IMAGE_WIDTH * IMAGE_HEIGHT * 3);

    // Light 3x3 Gaussian blur to smooth residual RC discretization
    auto getPixel = [&](int x, int y) -> GPU::Math::Vec3 {
        x = std::clamp(x, 0, IMAGE_WIDTH - 1);
        y = std::clamp(y, 0, IMAGE_HEIGHT - 1);
        const auto& p = pixels[y * IMAGE_WIDTH + x];
        return GPU::Math::Vec3(p.x, p.y, p.z);
    };

    const float kernel[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
    };

    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            GPU::Math::Vec3 c(0.0f, 0.0f, 0.0f);
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    c = c + getPixel(x + kx, y + ky) * kernel[ky + 1][kx + 1];
                }
            }
            c.x = std::sqrt(std::clamp(c.x, 0.0f, 1.0f));
            c.y = std::sqrt(std::clamp(c.y, 0.0f, 1.0f));
            c.z = std::sqrt(std::clamp(c.z, 0.0f, 1.0f));
            int dst = ((IMAGE_HEIGHT - 1 - y) * IMAGE_WIDTH + x) * 3;
            img[dst + 0] = static_cast<unsigned char>(256.0f * std::clamp(c.x, 0.0f, 0.999f));
            img[dst + 1] = static_cast<unsigned char>(256.0f * std::clamp(c.y, 0.0f, 0.999f));
            img[dst + 2] = static_cast<unsigned char>(256.0f * std::clamp(c.z, 0.0f, 0.999f));
        }
    }
    return stbi_write_png(filename, IMAGE_WIDTH, IMAGE_HEIGHT, 3, img.data(), IMAGE_WIDTH * 3) != 0;
}

} // namespace RC
