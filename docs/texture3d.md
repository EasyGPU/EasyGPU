# Texture3D

EasyGPU provides native `Texture3D` support for volumetric GPU compute, enabling read-write image operations on 3D data sets such as signed distance fields, voxel grids, and volume textures.

## Creating a Texture3D

```cpp
#include <GPU.h>
using namespace GPU::Runtime;

// Create an empty 8x8x8 RGBA8 volume
Texture3D<PixelFormat::RGBA8> volume(8, 8, 8);

// Create from initial data
std::vector<uint8_t> voxels(8 * 8 * 8 * 4, 255);
Texture3D<PixelFormat::RGBA8> volume2(8, 8, 8, voxels.data());
```

## Upload and Download

```cpp
// Upload entire volume
volume.Upload(voxels.data());

// Upload a sub-region
volume.UploadSubRegion(2, 2, 2, 4, 4, 4, subVoxels.data());

// Download entire volume
std::vector<uint8_t> result(8 * 8 * 8 * 4);
volume.Download(result.data());
```

## Using Texture3D in Kernels

Bind a `Texture3D` inside a kernel to get a `TextureRef3D`, then use `Read(x, y, z)` and `Write(x, y, z, color)`:

```cpp
Texture3D<PixelFormat::RGBA8> tex(8, 8, 8);

Kernel1D kernel([&](Var<int>& id) {
    auto vol = tex.Bind();

    Var<int> x = id % 8;
    Var<int> y = (id / 8) % 8;
    Var<int> z = id / 64;

    Var<Vec4> color = vol.Read(x, y, z);
    vol.Write(x, y, z, Vec4(1.0f) - color);
});

kernel.Dispatch(8, true);
```

## Supported Formats

All pixel formats available to `Texture2D` are also supported by `Texture3D`:

| Format | Bytes Per Voxel | Typical Use |
|:-------|:----------------|:------------|
| `RGBA8` | 4 | Color volumes, density fields |
| `RGBA32F` | 16 | High-precision vector fields |
| `R32F` | 4 | Signed distance fields, scalar density |
| `RG32F` | 8 | Velocity fields |
| `RGBA32I` | 16 | Integer label volumes |

## Type Aliases

```cpp
Texture3DRGBA8   = Texture3D<PixelFormat::RGBA8>;
Texture3DRGBA32F = Texture3D<PixelFormat::RGBA32F>;
Texture3DR32F    = Texture3D<PixelFormat::R32F>;
Texture3DRG32F   = Texture3D<PixelFormat::RG32F>;
Texture3DR8      = Texture3D<PixelFormat::R8>;
```

## Texture3DSlot (Dynamic Binding)

Switch between different 3D textures at runtime without recompiling:

```cpp
Texture3DSlot<RGBA8> volumeSlot;

Kernel1D kernel([&](Var<int>& id) {
    auto vol = volumeSlot.Bind();
    // ...
});

Texture3D<PixelFormat::RGBA8> volA(8, 8, 8);
Texture3D<PixelFormat::RGBA8> volB(16, 16, 16);

volumeSlot.Attach(volA);
kernel.Dispatch(8, true);

volumeSlot.Attach(volB);  // No recompilation!
kernel.Dispatch(64, true);
```

## Backend Support

`Texture3D` is implemented on both backends:

- **OpenGL**: Uses `GL_TEXTURE_3D` with `glTexImage3D` / `glTexSubImage3D`
- **Vulkan**: Uses `VK_IMAGE_TYPE_3D` with `vkCmdCopyBufferToImage`

## Sampling (Fragment Shader)

> **Deprecated path.** `FragmentKernel2D` is superseded by
> [GraphicsPipeline](graphics-pipeline.md). `BindSampler()` works identically
> in both APIs.

`Texture3D` also supports sampler binding for fragment shaders:

```cpp
Texture3D<PixelFormat::RGBA8> tex(64, 64, 64);

FragmentKernel2D kernel("VolumeSlice",
    [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
        auto sampler = tex.BindSampler();
        Float2 uv = fragCoord / resolution;
        Float3 uvw = MakeFloat3(uv.x(), uv.y(), 0.5f);
        fragColor = sampler.Sample(uvw);
    },
    512, 512
);
```
