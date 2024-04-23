#include <glad/glad.h>
#include <glfw3.h>
#include <iostream>
#include <vector>
#include <glm.hpp>
#include <iomanip>

#include "Shader.h"
#include "Shape.h"
#include "helper.h"
#include "BVH.h"
#include "Renderer.h"

#include "../vendor/hdrloader/hdrloader.h"


#define WIDTH 800
#define HEIGHT 800
#define PI 3.14159

using namespace GLRayTracing;
using namespace glm;

/***********Camera Parameters*************/
float theta = 0;
float phi = 0;
float radius = 5;
float rotationSpeed = 1;
float translateSpeed = 0.05;

//用作随机种子
unsigned int frameCounter = 0;

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        phi -= rotationSpeed;
        frameCounter = 0;
    }

    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        phi += rotationSpeed;
        frameCounter = 0;
    }

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        theta += rotationSpeed;
        theta = min(theta, 90.0f);
        frameCounter = 0;
    }

    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        theta -= rotationSpeed;
        theta = max(theta, -90.0f);
        frameCounter = 0;
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        radius -= translateSpeed;
        radius = max(radius, 0.0f);
        frameCounter = 0;
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        radius += translateSpeed;
        frameCounter = 0;
    }
}

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwSetWindowPos(window, 800, 400);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    /*****************finished initialization************/


    Shader pass1_shader = Shader("shader/vertex_shader.vs", "shader/pass1_shader.fs");
    Shader pass2_shader = Shader("shader/vertex_shader.vs", "shader/pass2_shader.fs");
    Shader hdr_shader = Shader("shader/vertex_shader.vs", "shader/test_hdr.fs");
    
    RenderPass pass1(WIDTH, HEIGHT, pass1_shader);
    unsigned int lastFrame = GetTextureRGB32F(pass1.m_Width, pass1.m_Height);
    unsigned int pass1Frame = GetTextureRGB32F(pass1.m_Width, pass1.m_Height);
    pass1.m_ColorAttachment = pass1Frame;
    pass1.BindData(false);

    RenderPass pass2(WIDTH, HEIGHT, pass2_shader);
    pass2.BindData(true);

    pass1.m_Shader.Use();
    pass1.m_Shader.SetInt("width", WIDTH);
    pass1.m_Shader.SetInt("height", HEIGHT);

    //hdr
    HDRLoaderResult hdrRes;
    unsigned int hdrMap = 0;
    unsigned int hdrCache = 0;
    int hdrResolution = 0;
    bool r = HDRLoader::load("hdr/sunflowers_puresky_4k.hdr", hdrRes);
    if (r)
    {
        hdrMap = GetTextureRGB32F(hdrRes.width, hdrRes.height);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, hdrRes.cols);
        std::cout << "hdr map loaded successfully" << std::endl;

        hdrResolution = hdrRes.width;

        // hdr 重要性采样 cache
        hdrCache = GetTextureRGB32F(hdrRes.width, hdrRes.height);
        float* cache = CalculateHdrCache(hdrRes.cols, hdrRes.width, hdrRes.height);
        //for (int i = 0; i < 512; i++) std::cout << cache[i] << std::endl;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, cache);
        std::cout << "hdr cache computed successfully" << std::endl;

    }

    //scene
    std::vector<Triangle> triangles;
    Material m;
    m.baseColor = vec3(0, 1, 1);
    m.roughness = 0.0;
    m.subsurface = 0.0;
    m.specular = 1.0;
    m.metallic = 0.0;
    m.clearcoat = 0.5;
    m.clearcoatGloss = 0.3;
    ReadObj("obj/bunny.obj", triangles, m, GetTransformMatrix(vec3(0, 0, 0), vec3(0.3, -1.6, -0.1), vec3(1.5, 1.5, 1.5)), true);

    m = Material();
    m.baseColor = vec3(0.1, 0.1, 0.1);
    m.roughness = 0.0;
    m.subsurface = 0.0;
    m.specular = 1.0;
    m.metallic = 0.0;
    m.clearcoat = 1.0;
    m.clearcoatGloss = 0.3;
    ReadObj("obj/quad.obj", triangles, m, GetTransformMatrix(vec3(0, 0, 0), vec3(0, -1.4, 0), vec3(10, 0.01, 10)), false);

    /*m.baseColor = vec3(1, 1, 0);
    ReadObj("obj/quad.obj", triangles, m, GetTransformMatrix(vec3(90, 90, 90), vec3(0.0, 0, -0.5), vec3(1, 1, 1)), false);

    m.baseColor = vec3(1, 0, 0);
    ReadObj("obj/quad.obj", triangles, m, GetTransformMatrix(vec3(0, 91, 0), vec3(-0.9, 0, 0), vec3(1, 1, 0.1)), false);*/

    /*m = Material();
    m.baseColor = vec3(1, 1, 1);
    m.emissive = vec3(200, 200, 200);
    ReadObj("obj/quad.obj", triangles, m, GetTransformMatrix(vec3(90, 90, 90), vec3(0.0, 0.9, 0.0), vec3(1, 0.1, 1)), false);*/

    int nTriangles = triangles.size();

    std::cout << "模型导入完成" << std::endl;

    // 建立 bvh， 这一步不能再编码triangle之后，因为这一步会改变triangle数组
    std::vector<BVHNode> nodes;
    BuildBVHwithSAH(triangles, nodes, 0, triangles.size() - 1, 8);
    int nNodes = nodes.size();

    std::cout << "BVH构建完成" << std::endl;

    //处理顶点
    std::vector<Triangle_encoded> triangles_encoded(nTriangles);
    for (int i = 0; i < nTriangles; i++)
    {
        triangles_encoded[i] = EncodeTriangle(triangles[i]);
    }

    //写入纹理
    GLuint trianglesTextureBuffer;
    GLuint tbo0;
    glGenBuffers(1, &tbo0);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo0);
    glBufferData(GL_TEXTURE_BUFFER, nTriangles * sizeof(Triangle_encoded), &triangles_encoded[0], GL_STATIC_DRAW);

    glGenTextures(1, &trianglesTextureBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo0);

    

    //处理bvh

    // 编码 BVHNode, aabb
    std::vector<BVHNode_encoded> nodes_encoded(nNodes);
    for (int i = 0; i < nNodes; i++)
    {
        nodes_encoded[i] = EncodeBVHNode(nodes[i]);
    }

    GLuint nodesTextureBuffer;
    // BVHNode 数组
    GLuint tbo1;
    glGenBuffers(1, &tbo1);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo1);
    glBufferData(GL_TEXTURE_BUFFER, nodes_encoded.size() * sizeof(BVHNode_encoded), &nodes_encoded[0], GL_STATIC_DRAW);
    glGenTextures(1, &nodesTextureBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo1);


    /***************Set Shader Sampler****************/
    /*
    * 要放在循环里初始化部分，因为纹理单元每次会修改
    * 但事实上，把lastPass的纹理存在texture0 这一部分就不需要放进循环
    * 根据具体情况，如果需要的纹理比较多，交替存进相同的纹理单元，就要注意重新设置纹理单元
    * 实际上一个纹理单元可以绑定不同的纹理类型。但相同类型只能有一个
    * 所以这里绑定texture0并不会发生冲突
    */
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);
    pass1.m_Shader.SetInt("triangles", 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
    pass1.m_Shader.SetInt("nodes", 1);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, hdrMap);
    pass1.m_Shader.SetInt("hdrMap", 3);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, hdrCache);
    pass1.m_Shader.SetInt("hdrCache", 4);


    /******************Set Shader uniform******************/
    pass1.m_Shader.SetInt("nTriangles", nTriangles);
    pass1.m_Shader.SetInt("nNodes", nNodes);
    pass1.m_Shader.SetInt("hdrResolution", hdrResolution);


    //loop
    clock_t t1 = clock(), t2;
    double dt, fps;

    while (!glfwWindowShouldClose(window))
    {
        t2 = clock();
        dt = (double)(t2 - t1) / CLOCKS_PER_SEC;
        fps = 1.0 / dt;
        std::cout << "\r";
        std::cout << std::fixed << std::setprecision(2) << "FPS : " << fps << "    迭代次数: " << frameCounter;
        t1 = t2;
        vec3 eye = vec3(radius * cos(theta * PI / 180) * sin(phi * PI / 180),
            radius * sin(radians(theta)),
            radius * cos(theta * PI / 180) * cos(phi * PI / 180)
            );

        mat4 cameraRotate = lookAt(eye, vec3(0, 0, 0), vec3(0, 1, 0));
        cameraRotate = inverse(cameraRotate);// lookat 的逆矩阵将光线方向进行转换

        pass1.m_Shader.Use();
        pass1.m_Shader.SetVec3("eye", eye);
        pass1.m_Shader.SetMat4("cameraRotate", cameraRotate);
        pass1.m_Shader.SetUint("frameCounter", frameCounter++);
        
        processInput(window);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, lastFrame);
        pass1.m_Shader.SetInt("lastFrame", 2);

        pass1.Draw();
        //保存一份帧缓冲的输出
        CopyTexture(lastFrame, pass1Frame, pass1.m_Width, pass1.m_Height);

        pass2.Draw(pass1Frame);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    glfwTerminate();
    return 0;
}

