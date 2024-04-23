#pragma once
#include <glad/glad.h>
#include <glm.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
namespace GLRayTracing {
	using namespace glm;
	class Shader
	{
	public:
		Shader(const char* vertexPath, const char* fragmentPath);
		void Use();

		void SetBool(const std::string& name, bool value) const;
		void SetInt(const std::string& name, int value) const;
		void SetUint(const std::string& name, unsigned int value) const;
		void SetFloat(const std::string& name, float value) const;
		void SetVec3(const std::string& name, vec3 value) const;
		void SetMat4(const std::string& name, mat4 value) const;

	private:
		void CheckCompileErrors(unsigned int shader, std::string type);

	public:
		unsigned int m_id;
	};
}

