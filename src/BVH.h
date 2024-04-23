#pragma once
#include <glm.hpp>
#include <vector>
#include <algorithm>

#include "Shape.h"

#define INF             114514.0
namespace GLRayTracing {
    using namespace glm;

    // BVH 树节点
    struct BVHNode {
        int left, right;    // 左右子树索引
        int n, index;       // 叶子节点信息               
        vec3 AA, BB;        // 碰撞盒
    };

    struct BVHNode_encoded {
        vec3 childs;        // (left, right, 保留)
        vec3 leafInfo;      // (n, index, 保留)
        vec3 AA, BB;
    };

    int BuildBVH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n);

    // SAH 优化构建 BVH
    int BuildBVHwithSAH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n);

    BVHNode_encoded EncodeBVHNode(const BVHNode& node);
}

