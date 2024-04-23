#pragma once
#include "Shader.h"
namespace GLRayTracing {
	class RenderPass
	{
	public:
		RenderPass(int width, int height, Shader& shader);
		~RenderPass() = default;

		void BindData(bool finalPass = false);
		void Draw(unsigned int attachment = 0);

	public:
		unsigned int m_FBO = 0;
		unsigned int m_VAO = 0;
		unsigned int m_VBO = 0;
		Shader m_Shader;
		unsigned int m_ColorAttachment = 0;
		unsigned int m_Width;
		unsigned int m_Height;
	};
}

