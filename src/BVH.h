#pragma once
#include <glm.hpp>
#include <vector>
#include <algorithm>

#include "Shape.h"

#define INF             114514.0
namespace GLRayTracing {
    using namespace glm;

    // BVH ���ڵ�
    struct BVHNode {
        int left, right;    // ������������
        int n, index;       // Ҷ�ӽڵ���Ϣ               
        vec3 AA, BB;        // ��ײ��
    };

    struct BVHNode_encoded {
        vec3 childs;        // (left, right, ����)
        vec3 leafInfo;      // (n, index, ����)
        vec3 AA, BB;
    };

    int BuildBVH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n);

    // SAH �Ż����� BVH
    int BuildBVHwithSAH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n);

    BVHNode_encoded EncodeBVHNode(const BVHNode& node);
}

