#include "Renderer.h"
#include <glad/glad.h>
#include <glm.hpp>
#include <vector>
namespace GLRayTracing {
	using namespace glm;

	RenderPass::RenderPass(int width, int height, Shader& shader)
		:m_Width(width), m_Height(height), m_Shader(shader)
	{
	}

	void RenderPass::BindData(bool finalPass)
	{
		//framebuffer
		if (!finalPass)
			glGenFramebuffers(1, &m_FBO);
		glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

		//vertex buffer
		glGenBuffers(1, &m_VBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
		std::vector<vec3> square = { vec3(-1, -1, 0), vec3(1, -1, 0), vec3(-1, 1, 0), vec3(1, 1, 0), vec3(-1, 1, 0), vec3(1, -1, 0) };
		glBufferData(GL_ARRAY_BUFFER, square.size() * sizeof(vec3), &square[0], GL_STATIC_DRAW);

		//vertex array
		glGenVertexArrays(1, &m_VAO);
		glBindVertexArray(m_VAO);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		//不是finalPass则生成纹理
		if (!finalPass)
		{
			assert(m_ColorAttachment != 0);

			glBindTexture(GL_TEXTURE_2D, m_ColorAttachment);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ColorAttachment, 0);

			glDrawBuffer(GL_COLOR_ATTACHMENT0);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

	}
	void RenderPass::Draw(unsigned int attachment)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
		glBindVertexArray(m_VAO);

		m_Shader.Use();

		if (attachment)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, attachment);
			m_Shader.SetInt("lastPass", 0);
		}

		glViewport(0, 0, m_Width, m_Height);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glBindVertexArray(0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glUseProgram(0);
		
	}
}