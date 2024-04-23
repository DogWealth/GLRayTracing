#pragma once
#include <glm.hpp>
namespace GLRayTracing {
    using namespace glm;

    struct Material {
        vec3 emissive = vec3(0, 0, 0);  // ��Ϊ��Դʱ�ķ�����ɫ
        vec3 baseColor = vec3(1, 1, 1);
        float subsurface = 0.0;
        float metallic = 0.0;
        float specular = 0.0;
        float specularTint = 0.0;
        float roughness = 0.0;
        float anisotropic = 0.0;
        float sheen = 0.0;
        float sheenTint = 0.0;
        float clearcoat = 0.0;
        float clearcoatGloss = 0.0;
        float IOR = 1.0;
        float transmission = 0.0;
    };

    struct Triangle {
        vec3 p1, p2, p3;    // ��������
        vec3 n1, n2, n3;    // ���㷨��
        Material material;  // ����
    };

    struct Triangle_encoded {
        vec3 p1, p2, p3;    // ��������
        vec3 n1, n2, n3;    // ���㷨��
        vec3 emissive;      // �Է������
        vec3 baseColor;     // ��ɫ
        vec3 param1;        // (subsurface, metallic, specular)
        vec3 param2;        // (specularTint, roughness, anisotropic)
        vec3 param3;        // (sheen, sheenTint, clearcoat)
        vec3 param4;        // (clearcoatGloss, IOR, transmission)
    };

    Triangle_encoded EncodeTriangle(const Triangle& t);


}

