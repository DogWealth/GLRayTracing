#pragma once
#include <glad/glad.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Shape.h"
namespace GLRayTracing {
    using namespace glm;
    // 读取 obj
    static void ReadObj(const std::string& filepath, std::vector<Triangle>& triangles, Material material, mat4 trans, bool smoothNormal) {

        // 顶点位置，索引
        std::vector<vec3> vertices;
        std::vector<GLuint> indices;

        // 打开文件流
        std::ifstream fin(filepath);
        std::string line;
        if (!fin.is_open()) {
            std::cout << "文件 " << filepath << " 打开失败" << std::endl;
            exit(-1);
        }

        // 计算 AABB 盒，归一化模型大小
        float maxx = -11451419.19;
        float maxy = -11451419.19;
        float maxz = -11451419.19;
        float minx = 11451419.19;
        float miny = 11451419.19;
        float minz = 11451419.19;

        // 按行读取
        while (std::getline(fin, line)) {
            std::istringstream sin(line);   // 以一行的数据作为 string stream 解析并且读取
            std::string type;
            GLfloat x, y, z;
            int v0, v1, v2;
            int vn0, vn1, vn2;
            int vt0, vt1, vt2;
            char slash;

            // 统计斜杆数目，用不同格式读取
            int slashCnt = 0;
            for (int i = 0; i < line.length(); i++) {
                if (line[i] == '/') slashCnt++;
            }

            // 读取obj文件
            sin >> type;
            if (type == "v") {
                sin >> x >> y >> z;
                vertices.push_back(vec3(x, y, z));
                maxx = max(maxx, x); maxy = max(maxx, y); maxz = max(maxx, z);
                minx = min(minx, x); miny = min(minx, y); minz = min(minx, z);
            }
            if (type == "f") {
                if (slashCnt == 6) {
                    sin >> v0 >> slash >> vt0 >> slash >> vn0;
                    sin >> v1 >> slash >> vt1 >> slash >> vn1;
                    sin >> v2 >> slash >> vt2 >> slash >> vn2;
                }
                else if (slashCnt == 3) {
                    sin >> v0 >> slash >> vt0;
                    sin >> v1 >> slash >> vt1;
                    sin >> v2 >> slash >> vt2;
                }
                else {
                    sin >> v0 >> v1 >> v2;
                }
                indices.push_back(v0 - 1);
                indices.push_back(v1 - 1);
                indices.push_back(v2 - 1);
            }
        }

        // 模型大小归一化
        float lenx = maxx - minx;
        float leny = maxy - miny;
        float lenz = maxz - minz;
        float maxaxis = max(lenx, max(leny, lenz));
        for (auto& v : vertices) {
            v.x /= maxaxis;
            v.y /= maxaxis;
            v.z /= maxaxis;
        }

        // 通过矩阵进行坐标变换
        for (auto& v : vertices) {
            vec4 vv = vec4(v.x, v.y, v.z, 1);
            vv = trans * vv;
            v = vec3(vv.x, vv.y, vv.z);
        }

        // 生成法线
        std::vector<vec3> normals(vertices.size(), vec3(0, 0, 0));
        for (int i = 0; i < indices.size(); i += 3) {
            vec3 p1 = vertices[indices[i]];
            vec3 p2 = vertices[indices[i + 1]];
            vec3 p3 = vertices[indices[i + 2]];
            vec3 n = normalize(cross(p2 - p1, p3 - p1));
            normals[indices[i]] += n;
            normals[indices[i + 1]] += n;
            normals[indices[i + 2]] += n;
        }

        // 构建 Triangle 对象数组
        int offset = triangles.size();  // 增量更新
        triangles.resize(offset + indices.size() / 3);
        for (int i = 0; i < indices.size(); i += 3) {
            Triangle& t = triangles[offset + i / 3];
            // 传顶点属性
            t.p1 = vertices[indices[i]];
            t.p2 = vertices[indices[i + 1]];
            t.p3 = vertices[indices[i + 2]];
            if (!smoothNormal) {
                vec3 n = normalize(cross(t.p2 - t.p1, t.p3 - t.p1));
                t.n1 = n; t.n2 = n; t.n3 = n;
            }
            else {
                t.n1 = normalize(normals[indices[i]]);
                t.n2 = normalize(normals[indices[i + 1]]);
                t.n3 = normalize(normals[indices[i + 2]]);
            }

            // 传材质
            t.material = material;
        }
    }

    // 模型变换矩阵
    static mat4 GetTransformMatrix(vec3 rotateCtrl, vec3 translateCtrl, vec3 scaleCtrl) {
        glm::mat4 unit(    // 单位矩阵
            glm::vec4(1, 0, 0, 0),
            glm::vec4(0, 1, 0, 0),
            glm::vec4(0, 0, 1, 0),
            glm::vec4(0, 0, 0, 1)
        );
        mat4 scale = glm::scale(unit, scaleCtrl);
        mat4 translate = glm::translate(unit, translateCtrl);
        mat4 rotate = unit;
        rotate = glm::rotate(rotate, glm::radians(rotateCtrl.x), glm::vec3(1, 0, 0));
        rotate = glm::rotate(rotate, glm::radians(rotateCtrl.y), glm::vec3(0, 1, 0));
        rotate = glm::rotate(rotate, glm::radians(rotateCtrl.z), glm::vec3(0, 0, 1));

        mat4 model = translate * rotate * scale;
        return model;
    }

    // 按照三角形中心排序 -- 比较函数
    static bool cmpx(const Triangle& t1, const Triangle& t2) {
        vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
        vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
        return center1.x < center2.x;
    }
    static bool cmpy(const Triangle& t1, const Triangle& t2) {
        vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
        vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
        return center1.y < center2.y;
    }
    static bool cmpz(const Triangle& t1, const Triangle& t2) {
        vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
        vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
        return center1.z < center2.z;
    }

    static GLuint GetTextureRGB32F(int width, int height) {
        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        return tex;
    }

    static void CopyTexture(unsigned int des, unsigned int src, int width, int height)
    {
        unsigned int fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, src, 0);
        glBindTexture(GL_TEXTURE_2D, des);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    //计算 HDR 贴图相关缓存信息
    static float* CalculateHdrCache(float* hdr, int width, int height)
    {
        float lumSum = 0.0f;

        // 初始化 h 行 w 列的概率密度 pdf 并 统计总亮度
        std::vector<std::vector<float>> pdf(height, std::vector<float>(width, 0.0f));

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                float R = hdr[3 * (i * width + j)];
                float G = hdr[3 * (i * width + j) + 1];
                float B = hdr[3 * (i * width + j) + 2];
                pdf[i][j] = 0.2 * R + 0.7 * G + 0.1 * B;

                lumSum += pdf[i][j];
            }
        }

        // 概率密度归一化
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
                pdf[i][j] /= lumSum;
            }

        // 累加每一列得到 x 的边缘概率密度
        std::vector<float> pdf_x_margin(width, 0.0f);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                pdf_x_margin[i] += pdf[j][i];
            }
        }

        // 计算 x 的边缘分布函数
        std::vector<float> cdf_x_margin = pdf_x_margin;
        for (int i = 1; i < width; i++)
            cdf_x_margin[i] += cdf_x_margin[i - 1];

        // 计算 y 在 X=x 下的条件概率密度函数
        std::vector<std::vector<float>> pdf_y_condition = pdf;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                pdf_y_condition[i][j] /= pdf_x_margin[j];
            }
        }

        // 计算 y 在 X=x 下的条件概率分布函数
        std::vector<std::vector<float>> cdf_y_condition = pdf_y_condition;
        for (int j = 0; j < width; j++)
        {
            for (int i = 1; i < height; i++)
            {
                cdf_y_condition[i][j] += cdf_y_condition[i - 1][j];
            }
        }

        // cdf_y_condiciton 转置为按列存储
        // cdf_y_condiciton[i] 表示 y 在 X=i 下的条件概率分布函数
        std::vector<std::vector<float>> temp = cdf_y_condition;
        cdf_y_condition = std::vector<std::vector<float>>(width, std::vector<float>(height));
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                cdf_y_condition[j][i] = temp[i][j];
            }
        }

        // 穷举 xi_1, xi_2 预计算样本 xy
        // sample_x[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 x
        // sample_y[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 y
        // sample_p[i][j] 表示取 (i, j) 点时的概率密度
        std::vector<std::vector<float>> sample_x(height, std::vector<float>(width));
        std::vector<std::vector<float>> sample_y(height, std::vector<float>(width));
        std::vector<std::vector<float>> sample_p(height, std::vector<float>(width));

        for(int i =0; i < height ;i++)
        {
            for (int j = 0; j < width; j++)
            {
                float xi_1 = float(i) / height;
                float xi_2 = float(j) / width;

                int x = std::lower_bound(cdf_x_margin.begin(), cdf_x_margin.end(), xi_1) -
                        cdf_x_margin.begin();

                int y = std::lower_bound(cdf_y_condition[x].begin(), cdf_y_condition[x].end(), xi_2) - cdf_y_condition[x].begin();

                sample_x[i][j] = float(x) / width;
                sample_y[i][j] = float(y) / height;
                sample_p[i][j] = pdf[i][j];
                //先采样的到坐标，用坐标再采样cache得到pdf。这里pdf不是对应x，y的，而是i，j
            }
        }

        // 整合结果到纹理
        // R,G 通道存储样本 (x,y) 而 B 通道存储 pdf(i, j)
        float* cache = new float[width * height * 3];
        //for (int i = 0; i < width * height * 3; i++) cache[i] = 0.0;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                cache[3 * (i * width + j)] = sample_x[i][j];        // R
                cache[3 * (i * width + j) + 1] = sample_y[i][j];    // G
                cache[3 * (i * width + j) + 2] = sample_p[i][j];    // B
            }
        }

        return cache;

    }
}