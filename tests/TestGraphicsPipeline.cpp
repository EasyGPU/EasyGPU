/**
 * @file TestGraphicsPipeline.cpp
 * @brief Tests for High #9: VK_KHR_dynamic_rendering Phase 1.
 */

#include <GPU.h>
#include <iostream>
#include <vector>

int main() {
	try {
		std::cout << "=== Graphics Pipeline Tests ===" << std::endl;

		GPU::Runtime::AutoInitContext();
		auto *backend = GPU::Runtime::Context::GetBackend();
		if (!backend || !backend->IsInitialized()) {
			std::cout << "Failed to initialize backend" << std::endl;
			return 1;
		}
		backend->MakeCurrent();

		if (!backend->GetCaps().supportsGraphics) {
			std::cout << "Graphics not supported, skipping all tests" << std::endl;
			backend->MakeNoneCurrent();
			return 0;
		}

		int			passed = 0, total = 0;

		const char *vsSrc =
			"#version 450\nvoid main() {\n\tvec2 pos;\n\tpos.x = float((gl_VertexIndex & 1) << 2) - 1.0;\n\tpos.y = "
			"float((gl_VertexIndex & 2) << 1) - 1.0;\n\tgl_Position = vec4(pos, 0.0, 1.0);\n}\n";
		const char *fsSrc = "#version 450\nlayout(location = 0) out vec4 outColor;\nvoid main() {\n\toutColor = "
							"vec4(0.5, 0.3, 0.1, 1.0);\n}\n";

		// Test 1: Create VS and FS shaders
		{
			GPU::Backend::ShaderDesc vsDesc{GPU::Backend::ShaderType::Vertex, vsSrc, "main"};
			auto					 vs = backend->CreateShader(vsDesc);
			if (vs == GPU::Backend::INVALID_SHADER_HANDLE) {
				std::cout << "[Test 1] FAIL: vertex shader creation failed" << std::endl;
				backend->MakeNoneCurrent();
				return 1;
			}

			GPU::Backend::ShaderDesc fsDesc{GPU::Backend::ShaderType::Fragment, fsSrc, "main"};
			auto					 fs = backend->CreateShader(fsDesc);
			if (fs == GPU::Backend::INVALID_SHADER_HANDLE) {
				backend->DestroyShader(vs);
				std::cout << "[Test 1] FAIL: fragment shader creation failed" << std::endl;
				backend->MakeNoneCurrent();
				return 1;
			}

			backend->DestroyShader(vs);
			backend->DestroyShader(fs);
			total++;
			passed++;
			std::cout << "[Test 1 CreateShaders] PASS" << std::endl;
		}

		// Test 2: Create graphics pipeline
		{
			GPU::Backend::ShaderDesc		   vsDesc{GPU::Backend::ShaderType::Vertex, vsSrc, "main"};
			auto							   vs = backend->CreateShader(vsDesc);
			GPU::Backend::ShaderDesc		   fsDesc{GPU::Backend::ShaderType::Fragment, fsSrc, "main"};
			auto							   fs = backend->CreateShader(fsDesc);

			GPU::Backend::GraphicsPipelineDesc pipeDesc;
			pipeDesc.vertexShader		   = vs;
			pipeDesc.fragmentShader		   = fs;
			pipeDesc.topology			   = GPU::Backend::PrimitiveTopology::TriangleList;
			pipeDesc.colorAttachmentFormat = GPU::Backend::PixelFormat::RGBA8;
			auto pipeline				   = backend->CreateGraphicsPipeline(pipeDesc);
			if (pipeline == GPU::Backend::INVALID_PIPELINE_HANDLE) {
				backend->DestroyShader(vs);
				backend->DestroyShader(fs);
				std::cout << "[Test 2] FAIL: pipeline creation failed" << std::endl;
				backend->MakeNoneCurrent();
				return 1;
			}

			backend->DestroyPipeline(pipeline);
			backend->DestroyShader(vs);
			backend->DestroyShader(fs);
			total++;
			passed++;
			std::cout << "[Test 2 CreateGraphicsPipeline] PASS" << std::endl;
		}

		// Test 3: Render fullscreen triangle to texture and verify pixels
		{
			const uint32_t					   W = 256, H = 256;

			GPU::Backend::ShaderDesc		   vsDesc{GPU::Backend::ShaderType::Vertex, vsSrc, "main"};
			auto							   vs = backend->CreateShader(vsDesc);
			GPU::Backend::ShaderDesc		   fsDesc{GPU::Backend::ShaderType::Fragment, fsSrc, "main"};
			auto							   fs = backend->CreateShader(fsDesc);

			GPU::Backend::GraphicsPipelineDesc pipeDesc;
			pipeDesc.vertexShader			   = vs;
			pipeDesc.fragmentShader			   = fs;
			pipeDesc.topology				   = GPU::Backend::PrimitiveTopology::TriangleList;
			pipeDesc.colorAttachmentFormat	   = GPU::Backend::PixelFormat::RGBA8;
			auto					  pipeline = backend->CreateGraphicsPipeline(pipeDesc);

			GPU::Backend::TextureDesc texDesc;
			texDesc.width						 = W;
			texDesc.height						 = H;
			texDesc.format						 = GPU::Backend::PixelFormat::RGBA8;
			auto							  rt = backend->CreateTexture(texDesc);

			GPU::Backend::RenderPassBeginDesc rpDesc;
			rpDesc.colorAttachment = rt;
			rpDesc.clearColor[0]   = 0.0f;
			rpDesc.clearColor[1]   = 0.0f;
			rpDesc.clearColor[2]   = 0.0f;
			rpDesc.clearColor[3]   = 1.0f;
			rpDesc.clearColorFlag  = true;

			backend->BeginRendering(rpDesc);
			backend->SetViewport(0, 0, W, H);
			backend->SetScissor(0, 0, W, H);
			backend->BindPipeline(pipeline);
			backend->Draw(3, 1, 0, 0);
			backend->EndRendering();
			backend->Finish();

			std::vector<uint8_t> pixels(W * H * 4);
			backend->DownloadTexture(rt, 0, 0, W, H, pixels.data());

			// FS outputs (0.5, 0.3, 0.1, 1.0) → (128, 77, 26, 255)
			bool ok = true;
			for (uint32_t y = 50; y < 200 && ok; y += 30) {
				for (uint32_t x = 50; x < 200 && ok; x += 30) {
					size_t idx = (y * W + x) * 4;
					if (pixels[idx + 0] < 120 || pixels[idx + 0] > 135)
						ok = false;
					if (pixels[idx + 1] < 70 || pixels[idx + 1] > 85)
						ok = false;
					if (pixels[idx + 2] < 20 || pixels[idx + 2] > 35)
						ok = false;
				}
			}
			if (!ok) {
				size_t mid = (100 * W + 128) * 4;
				std::cout << "[Test 3] FAIL: pixels at (128,100): R=" << (int)pixels[mid + 0]
						  << " G=" << (int)pixels[mid + 1] << " B=" << (int)pixels[mid + 2] << std::endl;
				backend->DestroyTexture(rt);
				backend->DestroyPipeline(pipeline);
				backend->DestroyShader(vs);
				backend->DestroyShader(fs);
				backend->MakeNoneCurrent();
				return 1;
			}

			backend->DestroyTexture(rt);
			backend->DestroyPipeline(pipeline);
			backend->DestroyShader(vs);
			backend->DestroyShader(fs);
			total++;
			passed++;
			std::cout << "[Test 3 RenderAndVerify] PASS" << std::endl;
		}

		// Test 4: Draw outside BeginRendering must throw
		{
			GPU::Backend::ShaderDesc		   vsDesc{GPU::Backend::ShaderType::Vertex, vsSrc, "main"};
			auto							   vs = backend->CreateShader(vsDesc);
			GPU::Backend::ShaderDesc		   fsDesc{GPU::Backend::ShaderType::Fragment, fsSrc, "main"};
			auto							   fs = backend->CreateShader(fsDesc);

			GPU::Backend::GraphicsPipelineDesc pipeDesc;
			pipeDesc.vertexShader		   = vs;
			pipeDesc.fragmentShader		   = fs;
			pipeDesc.topology			   = GPU::Backend::PrimitiveTopology::TriangleList;
			pipeDesc.colorAttachmentFormat = GPU::Backend::PixelFormat::RGBA8;
			auto pipeline				   = backend->CreateGraphicsPipeline(pipeDesc);

			backend->BindPipeline(pipeline);
			bool caught = false;
			try {
				backend->Draw(3, 1, 0, 0);
			} catch (const std::runtime_error &) {
				caught = true;
			}

			backend->DestroyPipeline(pipeline);
			backend->DestroyShader(vs);
			backend->DestroyShader(fs);

			if (!caught) {
				std::cout << "[Test 4] FAIL: should have thrown" << std::endl;
				backend->MakeNoneCurrent();
				return 1;
			}
			total++;
			passed++;
			std::cout << "[Test 4 DrawOutsideRenderPass] PASS" << std::endl;
		}

		// Test 5: Double BeginRendering must throw
		{
			GPU::Backend::TextureDesc texDesc;
			texDesc.width						  = 64;
			texDesc.height						  = 64;
			texDesc.format						  = GPU::Backend::PixelFormat::RGBA8;
			auto							  tex = backend->CreateTexture(texDesc);

			GPU::Backend::RenderPassBeginDesc rpDesc;
			rpDesc.colorAttachment = tex;
			rpDesc.clearColorFlag  = false;

			backend->BeginRendering(rpDesc);
			bool caught = false;
			try {
				backend->BeginRendering(rpDesc);
			} catch (const std::runtime_error &) {
				caught = true;
			}
			backend->EndRendering();
			backend->DestroyTexture(tex);

			if (!caught) {
				std::cout << "[Test 5] FAIL: should have thrown" << std::endl;
				backend->MakeNoneCurrent();
				return 1;
			}
			total++;
			passed++;
			std::cout << "[Test 5 DoubleBeginRendering] PASS" << std::endl;
		}

		// Test 6: Allocate and generate a complete mip chain
		{
			const uint32_t W = 64, H = 32;
			std::vector<uint8_t> pixels(W * H * 4, 255);

			GPU::Backend::TextureDesc texDesc;
			texDesc.width	  = W;
			texDesc.height	  = H;
			texDesc.mipLevels = 7;
			texDesc.format	  = GPU::Backend::PixelFormat::RGBA8;
			auto tex		  = backend->CreateTexture(texDesc);

			backend->UploadTexture(tex, 0, 0, W, H, pixels.data());
			backend->GenerateMipmaps(tex);
			backend->DestroyTexture(tex);

			total++;
			passed++;
			std::cout << "[Test 6 GenerateMipmaps] PASS" << std::endl;
		}

		backend->MakeNoneCurrent();

		std::cout << "\nResults: " << passed << "/" << total << " passed" << std::endl;
		return (passed == total) ? 0 : 1;

	} catch (const std::exception &e) {
		std::cerr << "EXCEPTION: " << e.what() << std::endl;
		return 2;
	} catch (...) {
		std::cerr << "UNKNOWN EXCEPTION" << std::endl;
		return 3;
	}
}
