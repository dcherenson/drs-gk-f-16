#version 330 core

in vec3 Normal;

uniform vec3 color;
uniform vec3 lightDir;

out vec4 FragColor;

void main() {
    float ambient = 0.25f;
    vec3 n = normalize(Normal);
    float diffuse = max(dot(normalize(lightDir), -n), 0.0f);
    FragColor = vec4(color * (ambient + 0.75f * diffuse), 1.0f);
}