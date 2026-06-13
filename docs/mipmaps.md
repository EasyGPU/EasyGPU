# Mipmaps

EasyGPU provides generated mipmap chains for `Texture2D`, trilinear sampled access, explicit mip-level sampling, and explicit-gradient sampling for texture atlases and other advanced rendering workflows.

## Creating a Mipmapped Texture

Pass `MipmapMode::Generate` when creating a `Texture2D`:

```cpp
#include <GPU.h>

using namespace GPU::Runtime;

Texture2D<PixelFormat::RGBA8> texture(
    1024,
    1024,
    MipmapMode::Generate
);
```

EasyGPU allocates the complete mip chain down to `1x1`. For example, a `1024x1024` texture has 11 mip levels.

```cpp
uint32_t levels = texture.GetMipLevels();  // 11
```

Mipmaps are opt-in. Textures created without `MipmapMode::Generate` keep one mip level and preserve the existing nearest-filtered sampling behavior.

## Upload and Regeneration

For a mipmapped texture, `Upload()` and `UploadSubRegion()` automatically regenerate the complete mip chain after updating level zero:

```cpp
Texture2D<PixelFormat::RGBA8> texture(1024, 1024, MipmapMode::Generate);

texture.Upload(pixels.data());                         // Upload + regenerate
texture.UploadSubRegion(32, 32, 64, 64, patch.data()); // Update + regenerate
```

You can also regenerate the chain explicitly:

```cpp
texture.GenerateMipmaps();
```

The initial-data constructor also generates mipmaps:

```cpp
Texture2D<PixelFormat::RGBA8> texture(
    width,
    height,
    pixels.data(),
    MipmapMode::Generate
);
```

## Sampling

Bind the texture as a sampler. Normal `Sample()` calls use implicit derivatives and automatically select mip levels:

```cpp
auto sampler = texture.BindSampler();
Float4 color = sampler.Sample(uv);
```

Mipmapped textures use linear filtering between texels and mip levels. Non-mipmapped textures retain nearest filtering.

### Explicit Mip Level

Use `SampleLevel()` to select a mip level directly:

```cpp
Float lod = 3.0f;
Float4 color = sampler.SampleLevel(uv, lod);
```

This generates GLSL `textureLod()`.

### Explicit Gradients

Use `SampleGrad()` when implicit derivatives are incorrect, especially after discontinuous UV operations such as `Fract()`:

```cpp
Float2 tiledUV = Fract(sourceUV);
Float2 dx = Ddx(sourceUV);
Float2 dy = Ddy(sourceUV);

Float4 color = sampler.SampleGrad(tiledUV, dx, dy);
```

This generates GLSL `textureGrad()`. The gradients control mip selection while the wrapped UV controls the sampled location.

## Screen-Space Derivatives

EasyGPU provides fragment-shader derivative functions:

```cpp
Float  dx = Ddx(value);
Float2 dy = Ddy(uv);
```

| Function | GLSL Equivalent | Description |
|:---------|:----------------|:------------|
| `Ddx(value)` | `dFdx(value)` | Horizontal screen-space derivative |
| `Ddy(value)` | `dFdy(value)` | Vertical screen-space derivative |

Derivatives are only valid in fragment shaders. Do not use them in compute or vertex shaders.

## Texture Atlas Sampling

Applying `Fract()` before normal implicit-derivative sampling can produce incorrect mip selection at tile boundaries. The wrapped coordinate jumps from approximately `1.0` to `0.0`, so the GPU observes a large derivative and may select an excessively small mip.

Preserve derivatives from the unwrapped UV and apply the atlas transform to both coordinates and gradients:

```cpp
Float2 tiled = Fract(sourceUV);
Float2 scale = MakeFloat2(atlasScaleX, atlasScaleY);
Float2 uv = MakeFloat2(
    atlasOffsetX + tiled.x() * atlasScaleX,
    atlasOffsetY + tiled.y() * atlasScaleY
);

Float2 dx = Ddx(sourceUV) * scale;
Float2 dy = Ddy(sourceUV) * scale;

Float4 color = sampler.SampleGrad(uv, dx, dy);
```

Atlas slots should also include gutters whose texels repeat the texture edge. This prevents linear and mip filtering from blending neighboring atlas entries.

## Low-Level Backend API

The low-level backend API exposes mip allocation and generation:

```cpp
Backend::TextureDesc desc;
desc.width      = 1024;
desc.height     = 1024;
desc.mipLevels  = 11;
desc.format     = Backend::PixelFormat::RGBA8;

auto texture = backend->CreateTexture(desc);
backend->UploadTexture(texture, 0, 0, 1024, 1024, pixels.data());
backend->GenerateMipmaps(texture);
```

`mipLevels` must not exceed the complete mip chain supported by the texture dimensions.

## Backend Notes

- **Vulkan:** Uses image blits to generate 2D mip chains. The format must support linear filtered blits.
- **Vulkan:** Level zero uses a storage-compatible image view; sampled access uses a separate view covering the complete mip chain.
- **OpenGL:** Uses `glGenerateMipmap`.
- **3D textures:** The high-level generated-mipmap API currently targets `Texture2D`. Vulkan mipmap generation rejects 3D textures.
- **Integer textures:** Keep using non-mipmapped nearest sampling. Linear mip generation and filtering are not valid for integer formats.

## See Also

- [API Reference](api-reference.md#textures)
- [Graphics Pipeline](graphics-pipeline.md)
- [Texture3D](texture3d.md)
