#include "Shape.h"
namespace GLRayTracing {
	Triangle_encoded EncodeTriangle(const Triangle& t)
	{
		const Material& m = t.material;
		Triangle_encoded te{};

		te.p1 = t.p1;
		te.p2 = t.p2;
		te.p3 = t.p3;

		te.n1 = t.n1;
		te.n2 = t.n2;
		te.n3 = t.n3;

		te.emissive = m.emissive;
		te.baseColor = m.baseColor;
		te.param1 = vec3(m.subsurface, m.metallic, m.specular);
		te.param2 = vec3(m.specularTint, m.roughness, m.anisotropic);
		te.param3 = vec3(m.sheen, m.sheenTint, m.clearcoat);
		te.param4 = vec3(m.clearcoatGloss, m.IOR, m.transmission);

		return te;
	}
}