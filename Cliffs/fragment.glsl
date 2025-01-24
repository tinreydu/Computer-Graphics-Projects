#version 300 es
precision highp float;
uniform vec4 color;

uniform vec3 lightdir;
uniform vec3 lightcolor;
uniform vec3 halfway;
out vec4 fragColor;
in vec3 vnormal;
void main() {
    vec3 n = normalize(vnormal);
    bool steep_cliff = n.y > 0.5;

    float lambert = max(dot(n, lightdir), 0.0);
    

    float red = steep_cliff ?  0.2 : 0.6;
    float green = steep_cliff ? 0.6 : 0.3;
    float blue = steep_cliff ? 0.1 : 0.3;

    float shine = steep_cliff ? 150.0 : 25.0;

    float blinn = pow(max(dot(n, halfway), 0.0), shine);

    fragColor = vec4(
        vec3(red, green, blue) * (lightcolor * lambert)
        +
        (lightcolor * blinn) * 1.0
    , color.a);
}