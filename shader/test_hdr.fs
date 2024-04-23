#version 330 core
in vec3 pix;
out vec4 FragColor;

uniform sampler2D hdrMap;

void main()
{
    vec3 color = texture2D(hdrMap, pix.xy * 0.5 + 0.5).rgb;
    color = color / (color + vec3(1, 1, 1));
    color = pow(color, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}