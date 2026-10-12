#version 330 core

in vec3 Normal;
in vec3 VertexColor;

uniform vec3 color;
uniform vec3 lightDir;

out vec4 FragColor;

void main() {
    vec3 n = normalize(Normal);
    float ambient = 0.28f;
    float diffuse = max(dot(normalize(lightDir), -n), 0.0f);
    vec3 base = VertexColor * color;
    FragColor = vec4(base * (ambient + 0.72f * diffuse), 1.0f);
}
