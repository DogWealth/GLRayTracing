#version 330 core
in vec3 pix;
out vec4 FragColor;


uniform samplerBuffer triangles;
uniform samplerBuffer nodes;

uniform sampler2D lastFrame;
uniform sampler2D hdrMap;
uniform sampler2D hdrCache;

uniform int nTriangles;
uniform int nNodes;
uniform int hdrResolution;

uniform uint frameCounter;
uniform int width;
uniform int height;

uniform vec3 eye;
uniform mat4 cameraRotate;

// ----------------------------------------------------------------------------- //

#define PI              3.1415926
#define INF             114514.0
#define SIZE_TRIANGLE   12
#define SIZE_BVHNODE    4

// ----------------------------------------------------------------------------- //

// Triangle 数据格式
struct Triangle {
    vec3 p1, p2, p3;    // 顶点坐标
    vec3 n1, n2, n3;    // 顶点法线
};

struct Material {
    vec3 emissive;          // 作为光源时的发光颜色
    vec3 baseColor;
    float subsurface;
    float metallic;
    float specular;
    float specularTint;
    float roughness;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    float IOR;
    float transmission;
};

// BVH 树节点
struct BVHNode {
    int left;           // 左子树
    int right;          // 右子树
    int n;              // 包含三角形数目
    int index;          // 三角形索引
    vec3 AA, BB;        // 碰撞盒
};

struct Ray {
    vec3 startPoint;
    vec3 direction;
};

struct HitResult
{
    bool isHit;
    bool isInside;
    float distance;
    vec3 hitPoint;
    vec3 normal;
    vec3 viewDir;
    Material material;

};

// 获取第 i 下标的三角形
Triangle getTriangle(int i) {
    int offset = i * SIZE_TRIANGLE;
    Triangle t;

    // 顶点坐标
    t.p1 = texelFetch(triangles, offset + 0).xyz;
    t.p2 = texelFetch(triangles, offset + 1).xyz;
    t.p3 = texelFetch(triangles, offset + 2).xyz;
    // 法线
    t.n1 = texelFetch(triangles, offset + 3).xyz;
    t.n2 = texelFetch(triangles, offset + 4).xyz;
    t.n3 = texelFetch(triangles, offset + 5).xyz;

    return t;
}

// 获取第 i 下标的三角形的材质
Material getMaterial(int i) {
    Material m;

    int offset = i * SIZE_TRIANGLE;
    vec3 param1 = texelFetch(triangles, offset + 8).xyz;
    vec3 param2 = texelFetch(triangles, offset + 9).xyz;
    vec3 param3 = texelFetch(triangles, offset + 10).xyz;
    vec3 param4 = texelFetch(triangles, offset + 11).xyz;
    
    m.emissive = texelFetch(triangles, offset + 6).xyz;
    m.baseColor = texelFetch(triangles, offset + 7).xyz;
    m.subsurface = param1.x;
    m.metallic = param1.y;
    m.specular = param1.z;
    m.specularTint = param2.x;
    m.roughness = param2.y;
    m.anisotropic = param2.z;
    m.sheen = param3.x;
    m.sheenTint = param3.y;
    m.clearcoat = param3.z;
    m.clearcoatGloss = param4.x;
    m.IOR = param4.y;
    m.transmission = param4.z;

    return m;
}

// 获取第 i 下标的 BVHNode 对象
BVHNode getBVHNode(int i) {
    BVHNode node;

    // 左右子树
    int offset = i * SIZE_BVHNODE;
    ivec3 childs = ivec3(texelFetch(nodes, offset + 0).xyz);
    ivec3 leafInfo = ivec3(texelFetch(nodes, offset + 1).xyz);
    node.left = int(childs.x);
    node.right = int(childs.y);
    node.n = int(leafInfo.x);
    node.index = int(leafInfo.y);

    // 包围盒
    node.AA = texelFetch(nodes, offset + 2).xyz;
    node.BB = texelFetch(nodes, offset + 3).xyz;

    return node;
}

// 光线和三角形求交 
HitResult hitTriangle(Triangle triangle, Ray ray) 
{
    HitResult res;
    res.distance = INF;
    res.isHit = false;
    res.isInside = false;

    vec3 p1 = triangle.p1;
    vec3 p2 = triangle.p2;
    vec3 p3 = triangle.p3;

    vec3 S = ray.startPoint;    // 射线起点
    vec3 d = ray.direction;     // 射线方向
    vec3 N = normalize(cross(p2-p1, p3-p1));    // 法向量

    // 从三角形背后（模型内部）击中
    if (dot(N, d) > 0.0f) {
        N = -N;   
        res.isInside = true;
    }

    // 如果视线和三角形平行
    if (abs(dot(N, d)) < 0.00001f) return res;

    // 距离
    float t = (dot(N, p1) - dot(S, N)) / dot(d, N);
    if (t < 0.0005f) return res;    // 如果三角形在光线背面

    // 交点计算
    vec3 P = S + d * t;

    // 判断交点是否在三角形中
    vec3 c1 = cross(p2 - p1, P - p1);
    vec3 c2 = cross(p3 - p2, P - p2);
    vec3 c3 = cross(p1 - p3, P - p3);
    bool r1 = (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0);
    bool r2 = (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0);

    // 命中，封装返回结果
    if (r1 || r2) {
        res.isHit = true;
        res.hitPoint = P;
        res.distance = t;
        res.normal = N;
        res.viewDir = d;
        // 根据交点位置插值顶点法线
        float alpha = (-(P.x-p2.x)*(p3.y-p2.y) + (P.y-p2.y)*(p3.x-p2.x)) / (-(p1.x-p2.x-0.00005)*(p3.y-p2.y+0.00005) + (p1.y-p2.y+0.00005)*(p3.x-p2.x+0.00005));
        float beta  = (-(P.x-p3.x)*(p1.y-p3.y) + (P.y-p3.y)*(p1.x-p3.x)) / (-(p2.x-p3.x-0.00005)*(p1.y-p3.y+0.00005) + (p2.y-p3.y+0.00005)*(p1.x-p3.x+0.00005));
        float gama  = 1.0 - alpha - beta;
        vec3 Nsmooth = alpha * triangle.n1 + beta * triangle.n2 + gama * triangle.n3;
        Nsmooth = normalize(Nsmooth);
        res.normal = (res.isInside) ? (-Nsmooth) : (Nsmooth);
    }

    return res;
}

// 暴力遍历数组下标范围 [l, r] 求最近交点
HitResult hitArray(Ray ray, int l, int r) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;
    for(int i=l; i<=r; i++) {
        Triangle triangle = getTriangle(i);
        HitResult r = hitTriangle(triangle, ray);
        if(r.isHit && r.distance<res.distance) {
            res = r;
            res.material = getMaterial(i);
        }
    }
    return res;
}

bool hitAABB(Ray ray, vec3 AA, vec3 BB)
{
    vec3 dir = ray.direction;
    vec3 p = ray.startPoint;

    float t = 0;

    if(dir.x != 0)
    {
        if(dir.x > 0)
            t = (AA.x - p.x) / dir.x;
        else
            t = (BB.x - p.x) / dir.x;

        if(t > 0)
        {
            vec3 hitP = p + t * dir;
            if(hitP.y < BB.y && hitP.y > AA.y && hitP.z < BB.z && hitP.z > AA.z)
                return true;
        }
    }

    if(dir.y != 0)
    {
        if(dir.y > 0)
            t = (AA.y - p.y) / dir.y;
        else
            t = (BB.y - p.y) / dir.y;

        if(t > 0)
        {
            vec3 hitP = p + t * dir;
            if(hitP.x < BB.x && hitP.x > AA.x && hitP.z < BB.z && hitP.z > AA.z)
                return true;
        }
    }

    if(dir.z != 0)
    {
        if(dir.z > 0)
            t = (AA.z - p.z) / dir.z;
        else
            t = (BB.z - p.z) / dir.z;

        if(t > 0)
        {
            vec3 hitP = p + t * dir;
            if(hitP.x < BB.x && hitP.x > AA.x && hitP.y < BB.y && hitP.y > AA.y)
                return true;
        }
    }

    return false;
}

//遍历bvh求交
HitResult hitBVH(Ray ray)
{
    HitResult res;
    res.isHit = false;
    res.distance = INF;

    //数组模拟栈
    int stack[256];
    int sp = 0;

    stack[sp++] = 0;

    while(sp > 0)
    {
        int top = stack[--sp];
        BVHNode node = getBVHNode(top);

        if(node.n > 0)
        {
            int left = node.index;
            int right = node.index + node.n - 1;
            HitResult r = hitArray(ray, left, right);

            if(r.isHit && r.distance < res.distance) res = r;
            continue;//若判断了hitAABB返回距离，这里可以直接return么？
        }

        bool b1 = false;
        bool b2 = false;

        if(node.left > 0)
        {
            BVHNode leftNode = getBVHNode(node.left);
            b1 = hitAABB(ray, leftNode.AA, leftNode.BB);
        }

        if(node.right > 0)
        {
            BVHNode rightNode = getBVHNode(node.right);
            b2 = hitAABB(ray, rightNode.AA, rightNode.BB);
        }

        if(b1) stack[sp++] = node.left;
        if(b2) stack[sp++] = node.right;
    }

    return res;
}

/*
helper function
*/

uint seed = uint(
    uint((pix.x * 0.5 + 0.5) * width)  * uint(1973) + 
    uint((pix.y * 0.5 + 0.5) * height) * uint(9277) + 
    uint(frameCounter) * uint(26699)) | uint(1);

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

vec2 CranleyPattersonRotation(vec2 p) {
    uint pseed = uint(
        uint((pix.x * 0.5 + 0.5) * width)  * uint(1973) + 
        uint((pix.y * 0.5 + 0.5) * height) * uint(9277) + 
        uint(114514/1919) * uint(26699)) | uint(1);
    
    float u = float(wang_hash(pseed)) / 4294967296.0;
    float v = float(wang_hash(pseed)) / 4294967296.0;

    p.x += u;
    if(p.x>1) p.x -= 1;
    if(p.x<0) p.x += 1;

    p.y += v;
    if(p.y>1) p.y -= 1;
    if(p.y<0) p.y += 1;

    return p;
}

float rand() {
    return float(wang_hash(seed)) / 4294967296.0;
}

// 半球均匀采样
vec3 SampleHemisphere(vec3 N) {
    float z = rand();
    float r = max(0, sqrt(1.0 - z*z));
    float phi = 2.0 * PI * rand();

    vec3 H;
    H.x = r * cos(phi);
    H.y = r * sin(phi);
    H.z = z;
    
    vec3 up = (abs(N.x) < 0.999) ? vec3(1, 0, 0) : vec3(0, 0, 1);
    vec3 tangent = normalize(cross(N, up));
    vec3 bitangent = normalize(cross(N, tangent));

    return H.x * tangent + H.y * bitangent + H.z * N;
}

vec3 SampleHemisphere(vec3 N, float r1, float r2) {
    float z = r1;
    float r = max(0, sqrt(1.0 - z*z));
    float phi = 2.0 * PI * r2;

    vec3 H;
    H.x = r * cos(phi);
    H.y = r * sin(phi);
    H.z = z;
    
    vec3 up = (abs(N.x) <= 0.999) ? vec3(1, 0, 0) : vec3(0, 0, 1);
    vec3 tangent = normalize(cross(N, up));
    vec3 bitangent = normalize(cross(N, tangent));

    return H.x * tangent + H.y * bitangent + H.z * N;
}


/****************hdr map*******************/
// 将三维向量 v 转为 HDR map 的纹理坐标 uv
vec2 SampleSphericalMap(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv /= vec2(2.0 * PI, PI);
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

// 获取 HDR 环境颜色
vec3 sampleHdr(vec3 v) {
    vec2 uv = SampleSphericalMap(normalize(v));
    vec3 color = texture2D(hdrMap, uv).rgb;
    //color = min(color, vec3(10));
    return color;
}


/************* B R D F *******************/
float SchlickFresnel(float u) {
    float m = clamp(1-u, 0, 1);
    float m2 = m*m;
    return m2*m2*m;
}

float GTR1(float NdotH, float a) {
    if (a >= 1) return 1/PI;
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return (a2-1) / (PI*log(a2)*t);
}

float GTR2(float NdotH, float a) {
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return a2 / (PI * t*t);
}

float smithG_GGX(float NdotV, float alphaG) {
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1 / (NdotV + sqrt(a + b - a*b));
}

//重要性采样
vec3 ImportanceSampleGTR2(vec3 N, vec3 V, float roughness, float r1, float r2) {
    float alpha = max(0.001, roughness * roughness);
    float phi_h = 2.0 * PI * r1;
    float cosTheta_h = sqrt((1.0 - r2)/(1.0 + (alpha * alpha - 1.0) * r2));
    float sinTheta_h = max(0, sqrt(1 - cosTheta_h * cosTheta_h));

    vec3 H;
    H.x = sinTheta_h * cos(phi_h);
    H.y = sinTheta_h * sin(phi_h);
    H.z = cosTheta_h;
    
    vec3 up = (abs(N.x) < 0.999) ? vec3(1, 0, 0) : vec3(0, 0, 1);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = normalize(cross(N, tangent));

    H = H.x * tangent + H.y * bitangent + H.z * N;

    vec3 L = reflect(-V, H);
    return L;
}

vec3 ImportanceSampleGTR1(vec3 N, vec3 V, float clearcoatGloss, float r1, float r2)
{
    float alpha = mix(0.1, 0.001, clearcoatGloss);
    float phi_h = 2.0 * PI * r1;
    float cosTheta_h = sqrt((1.0 - pow(alpha * alpha, 1 - r2)) / (1 - alpha * alpha));
    float sinTheta_h = max(0, sqrt(1 - cosTheta_h * cosTheta_h));

    float test = (1 - alpha * alpha);

    vec3 H;
    H.x = sinTheta_h * cos(phi_h);
    H.y = sinTheta_h * sin(phi_h);
    H.z = cosTheta_h;
    
    vec3 up = (abs(N.x) <= 0.999) ? vec3(1, 0, 0) : vec3(0, 0, 1);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = normalize(cross(N, tangent));

    H = H.x * tangent + H.y * bitangent + H.z * N;

    vec3 L = reflect(-V, H);

    
    return L;
}

vec3 ImportanceSampleHDR(float r1, float r2)
{
    vec2 xy = texture2D(hdrCache, vec2(r1, r2)).rg;
    xy.y = 1 - xy.y;

    float phi = 2 * PI * (xy.x - 0.5);
    float theta = PI * (xy.y - 0.5);

    vec3 L = vec3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi));
    return L;

}

//混合采样gtr1 + gtr2 + Hemisphere
vec3 BRDFSample(vec3 N, vec3 V, Material material, float r1, float r2, float r3)
{
    // 辐射度统计
    float r_diffuse = (1.0 - material.metallic);
    float r_specular = 1.0;
    float r_clearcoat = 0.25 * material.clearcoat;
    float r_sum = r_diffuse + r_specular + r_clearcoat;

    // 根据辐射度计算概率
    float p_diffuse = r_diffuse / r_sum;
    float p_specular = r_specular / r_sum;
    float p_clearcoat = r_clearcoat / r_sum;

    float rd = r3;

    //diffuse
    if(rd <= p_diffuse) return SampleHemisphere(N, r1, r2);

    //specular
    if(rd <= (p_specular + p_diffuse)) return ImportanceSampleGTR2(N, V, material.roughness , r1, r2);

    return ImportanceSampleGTR1(N, V, material.clearcoat, r1, r2);

}

// 获取 BRDF 在 L 方向上的概率密度
float BRDF_PDF(vec3 N, vec3 V, vec3 L, Material material)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if(NdotL < 0 || NdotV < 0) return 0;

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    // 镜面反射 -- 各向同性
    float alpha = max(0.001, material.roughness * material.roughness);
    float Ds = GTR2(NdotH, alpha); 
    float Dr = GTR1(NdotH, mix(0.1, 0.001, material.clearcoatGloss));   // 清漆

    // 分别计算三种 BRDF 的概率密度
    float pdf_diffuse = NdotL / PI;
    float pdf_specular = Ds * NdotH / (4.0 * dot(L, H));
    float pdf_clearcoat = Dr * NdotH / (4.0 * dot(L, H));

    // 辐射度统计
    float r_diffuse = (1.0 - material.metallic);
    float r_specular = 1.0;
    float r_clearcoat = 0.25 * material.clearcoat;
    float r_sum = r_diffuse + r_specular + r_clearcoat;

     // 根据辐射度计算选择某种采样方式的概率
    float p_diffuse = r_diffuse / r_sum;
    float p_specular = r_specular / r_sum;
    float p_clearcoat = r_clearcoat / r_sum;

    // 根据概率混合 pdf
    float pdf = p_diffuse   * pdf_diffuse 
              + p_specular  * pdf_specular
              + p_clearcoat * pdf_clearcoat;


    return pdf;
}

float HDR_PDF(vec3 L)
{
    vec2 uv = SampleSphericalMap(normalize(L));   // 方向向量转 uv 纹理坐标

    float pdf = texture2D(hdrCache, uv).b;      // 采样概率密度
    float theta = PI * (0.5 - uv.y);            // theta 范围 [-pi/2 ~ pi/2]
    float sin_theta = max(sin(theta), 1e-10);

    // 球坐标和图片积分域的转换系数
    float p_convert = float(hdrResolution * hdrResolution / 2) / (2.0 * PI * PI * sin_theta);  
    
    return pdf * p_convert;
}


/*
因为 pdf 涉及除法，就不得不考虑除数等于 0 的情况 
稍有不慎就会抛出 divided by zero 的 exception 从而 down 掉你的 fragment shader 
同时屏幕出现黑点，这些黑点永远不会消除，直到你关闭程序
*/
vec3 brdfEvaluate(vec3 V, vec3 N, vec3 L, in Material material)
{
    float NdotV = dot(V, N);
    float NdotL = dot(N, L);
    if(NdotL <= 0 || NdotV <=0) return vec3(0);

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    vec3 baseColor = material.baseColor;
    float metallic = material.metallic;
    float roughness = material.roughness;
    float subsurface = material.subsurface;
    float specularTint = material.specularTint;
    float specular = material.specular;

    //diffuse
    float Fd90 = 0.5 * 2 * roughness * LdotH * LdotH;
    float Fl = SchlickFresnel(NdotL);
    float Fv = SchlickFresnel(NdotV);
    float Fd = mix(1.0, Fd90, Fl) * mix(1.0, Fd90, Fv);

    // 次表面散射
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = mix(1.0, Fss90, Fl) * mix(1.0, Fss90, Fv);
    float ss = 1.25 * (Fss * (1.0 / (NdotL + NdotV) - 0.5) + 0.5);

    vec3 diffuse = baseColor * mix(Fd, ss, subsurface) / PI;

    //specular
    float Cdlum = 0.3 * baseColor.r + 0.6 * baseColor.g + 0.1 * baseColor.b;
    vec3 Ctint = (Cdlum > 0) ? (baseColor/Cdlum) : (vec3(1));  
    vec3 Cspec = specular * mix(vec3(1), Ctint, specularTint);
    vec3 F0 = mix(0.08*Cspec, baseColor, metallic); // 0° 镜面反射颜色

    float alpha = max(0.001, roughness * roughness);
    float Ds = GTR2(NdotH, alpha);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(F0, vec3(1), FH);
    float Gs = smithG_GGX(NdotL, roughness) * smithG_GGX(NdotV, roughness);

    vec3 spec = Ds * Fs * Gs;

    // 清漆
	float Dr = GTR1(NdotH, mix(0.1, 0.001, material.clearcoatGloss));
	float Fr = mix(0.04, 1.0, FH);
	float Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25);
	vec3 clearcoat = vec3(0.25 * Gr * Fr * Dr * material.clearcoat);

    vec3 color = diffuse * (1 - metallic) + spec + clearcoat;

    return color;

}

// 1 ~ 8 维的 sobol 生成矩阵
const uint V[8*32] = uint[](
    2147483648u,1073741824u,536870912u,268435456u,134217728u,67108864u,33554432u,16777216u,8388608u,4194304u,2097152u,1048576u,524288u,262144u,131072u,65536u,32768u,16384u,8192u,4096u,2048u,1024u,512u,256u,128u,64u,32u,16u,8u,4u,2u,1u,2147483648u,3221225472u,2684354560u,4026531840u,2281701376u,3422552064u,2852126720u,4278190080u,2155872256u,3233808384u,2694840320u,4042260480u,2290614272u,3435921408u,2863267840u,4294901760u,2147516416u,3221274624u,2684395520u,4026593280u,2281736192u,3422604288u,2852170240u,4278255360u,2155905152u,3233857728u,2694881440u,4042322160u,2290649224u,3435973836u,2863311530u,4294967295u,2147483648u,3221225472u,1610612736u,2415919104u,3892314112u,1543503872u,2382364672u,3305111552u,1753219072u,2629828608u,3999268864u,1435500544u,2154299392u,3231449088u,1626210304u,2421489664u,3900735488u,1556135936u,2388680704u,3314585600u,1751705600u,2627492864u,4008611328u,1431684352u,2147543168u,3221249216u,1610649184u,2415969680u,3892340840u,1543543964u,2382425838u,3305133397u,2147483648u,3221225472u,536870912u,1342177280u,4160749568u,1946157056u,2717908992u,2466250752u,3632267264u,624951296u,1507852288u,3872391168u,2013790208u,3020685312u,2181169152u,3271884800u,546275328u,1363623936u,4226424832u,1977167872u,2693105664u,2437829632u,3689389568u,635137280u,1484783744u,3846176960u,2044723232u,3067084880u,2148008184u,3222012020u,537002146u,1342505107u,2147483648u,1073741824u,536870912u,2952790016u,4160749568u,3690987520u,2046820352u,2634022912u,1518338048u,801112064u,2707423232u,4038066176u,3666345984u,1875116032u,2170683392u,1085997056u,579305472u,3016343552u,4217741312u,3719483392u,2013407232u,2617981952u,1510979072u,755882752u,2726789248u,4090085440u,3680870432u,1840435376u,2147625208u,1074478300u,537900666u,2953698205u,2147483648u,1073741824u,1610612736u,805306368u,2818572288u,335544320u,2113929216u,3472883712u,2290089984u,3829399552u,3059744768u,1127219200u,3089629184u,4199809024u,3567124480u,1891565568u,394297344u,3988799488u,920674304u,4193267712u,2950604800u,3977188352u,3250028032u,129093376u,2231568512u,2963678272u,4281226848u,432124720u,803643432u,1633613396u,2672665246u,3170194367u,2147483648u,3221225472u,2684354560u,3489660928u,1476395008u,2483027968u,1040187392u,3808428032u,3196059648u,599785472u,505413632u,4077912064u,1182269440u,1736704000u,2017853440u,2221342720u,3329785856u,2810494976u,3628507136u,1416089600u,2658719744u,864310272u,3863387648u,3076993792u,553150080u,272922560u,4167467040u,1148698640u,1719673080u,2009075780u,2149644390u,3222291575u,2147483648u,1073741824u,2684354560u,1342177280u,2281701376u,1946157056u,436207616u,2566914048u,2625634304u,3208642560u,2720006144u,2098200576u,111673344u,2354315264u,3464626176u,4027383808u,2886631424u,3770826752u,1691164672u,3357462528u,1993345024u,3752330240u,873073152u,2870150400u,1700563072u,87021376u,1097028000u,1222351248u,1560027592u,2977959924u,23268898u,437609937u
);

// 格林码 
uint grayCode(uint i) {
	return i ^ (i>>1);
}

// 生成第 d 维度的第 i 个 sobol 数
float sobol(uint d, uint i) {
    uint result = 0u;
    uint offset = d * 32u;
    for(uint j = 0u; i > 0u; i >>= 1, j++) 
        if(bool(i & 1u))
            result ^= V[j+offset];

    return float(result) * (1.0f/float(0xFFFFFFFFU));
}

// 生成第 i 帧的第 b 次反弹需要的二维随机向量
vec2 sobolVec2(uint i, uint b) {
    float u = sobol(b * 2u, grayCode(i));
    float v = sobol(b * 2u + 1u, grayCode(i));
    return vec2(u, v);
}

/*
pathTracing
*/

float misMixWeight(float a, float b) {
    float t = a * a;
    return t / (b*b + t);
}

vec3 pathTracing(HitResult res, int maxBounce)
{
    vec3 Lo = vec3(1);
    vec3 final_Lo = vec3(0);

    for(int bounce = 0; bounce < maxBounce; bounce++)
    {
        vec3 N = res.normal;
        vec3 V = -res.viewDir;

        // HDR 环境贴图重要性采样    
        Ray hdrTestRay;
        hdrTestRay.startPoint = res.hitPoint;
        hdrTestRay.direction = ImportanceSampleHDR(rand(), rand());

        // 进行一次求交测试 判断是否有遮挡
        if(dot(N, hdrTestRay.direction) > 0.0) { // 如果采样方向背向点 p 则放弃测试, 因为 N dot L < 0            
            HitResult hdrHit = hitBVH(hdrTestRay);
            
            // 天空光仅在没有遮挡的情况下积累亮度
            if(!hdrHit.isHit) {
                // 获取采样方向 L 上的: 1.光照贡献, 2.环境贴图在该位置的 pdf, 3.BRDF 函数值, 4.BRDF 在该方向的 pdf
                vec3 L = hdrTestRay.direction;
                vec3 color = sampleHdr(L);
                float pdf_light = HDR_PDF(L);
                vec3 f_r = brdfEvaluate(V, N, L, res.material);
                float pdf_brdf = BRDF_PDF(V, N, L, res.material);
                
                // 多重重要性采样
                float mis_weight = misMixWeight(pdf_light, pdf_brdf);
                //final_Lo += mis_weight * Lo * color * f_r * dot(N, L) / pdf_light;
                //final_Lo += Lo * color * f_r * dot(N, L) / pdf_light;
            }
        }

        vec2 uv = sobolVec2(frameCounter, uint(bounce));
        uv = CranleyPattersonRotation(uv);
        vec3 wi = BRDFSample(N, V, res.material, uv.x, uv.y, rand());
        
        vec3 L = wi;

        Ray randomRay;
        randomRay.direction = L;
        randomRay.startPoint = res.hitPoint;

        HitResult newRes = hitBVH(randomRay);

        float cosin = max(0, dot(L, N));
        vec3 f_r = brdfEvaluate(V, N, L, res.material);
        float pdf_brdf = BRDF_PDF(N, V, L, res.material);
        if(pdf_brdf <= 0) break;

        if(!newRes.isHit)
        {
            vec3 Le = sampleHdr(randomRay.direction);
            float pdf_light = HDR_PDF(randomRay.direction);
                
            // 多重重要性采样
            float mis_weight = misMixWeight(pdf_brdf, pdf_light);
            final_Lo += mis_weight * Lo * Le * f_r * cosin / pdf_brdf;

            //final_Lo += Lo * Le * f_r * cosin / pdf_light;
            break;
        }

        vec3 Le = newRes.material.emissive;

        Lo *= f_r * cosin / pdf_brdf;
        final_Lo += Lo * Le;

        res = newRes;
    }

    return final_Lo;
}


void main()
{
    Ray ray;
    vec2 AA = vec2((rand()-0.5)/float(width), (rand()-0.5)/float(height));

    ray.startPoint = eye;
    vec4 dir = cameraRotate * vec4(pix.xy + AA, -1.5, 0.0);
    ray.direction = normalize(dir.xyz);

    vec3 color = vec3(0, 0, 0);
    

    HitResult res = hitBVH(ray);
    if(res.isHit)
    {
        vec3 Le = res.material.emissive;
        int maxBounce = 4;

        vec3 Li = pathTracing(res, maxBounce);

        color = Le + Li;

    }
    else
    {
        color = sampleHdr(ray.direction);
    }

    vec3 lastColor = texture2D(lastFrame, pix.xy * 0.5 + 0.5).rgb;
    color = mix(lastColor, color, 1.0 / float(frameCounter + 1.0f));
    FragColor = vec4(color, 1);
} 