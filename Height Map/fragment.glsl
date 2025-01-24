#version 300 es
precision highp float;
uniform vec4 color;

uniform vec3 lightdir;
uniform vec3 lightcolor;
uniform vec3 halfway;

uniform float maxY;
uniform float minY;
out vec4 fragColor;
in vec3 vnormal;
in float outY;


vec3 rainbow(float y) {
    float norm_height = (y - minY) / (maxY - minY);
    vec3 color1, color2;
    float t;
    if (norm_height < 0.14) {
        color1 = color2 = vec3(0.58, 0.0, 0.83);
        t = 0.0;
    } else if (norm_height < 0.29 ) {
        color1 = vec3(0.58, 0.0, 0.83);
        color2 = vec3(0.29, 0.0, 0.51);
        t = (norm_height - 0.14) / (0.29-0.14);
    } else if (norm_height < 0.43 ) {
        color1 = vec3(0.29, 0.0, 0.51);
        color2 = vec3(0.0, 0.0, 1.0);
        t = (norm_height - 0.29) / (0.43-0.29);
    } else if (norm_height < 0.57 ) {
        color1 = vec3(0.0, 0.0, 1.0); 
        color2 = vec3(0.0, 1.0, 0.0);
        t = (norm_height - 0.43) / (0.57 - 0.43);
    } else if (norm_height < 0.71) {
        color1 = vec3(0.0, 1.0, 0.0); 
        color2 = vec3(1.0, 1.0, 0.0); 
        t = (norm_height - 0.57) / (0.71 - 0.57);
    } else if (norm_height < 0.86) {
        color1 = vec3(1.0, 1.0, 0.0); 
        color2 = vec3(1.0, 0.5, 0.0);
        t = (norm_height - 0.71) / (0.86 - 0.71);
    } else {
        color1 = vec3(1.0, 0.5, 0.0); 
        color2 = vec3(1.0, 0.0, 0.0); 
        t = (norm_height - 0.86) / (1.0 - 0.86);
    }
    return (1.0 - t) * color1 + (t * color2);
}
void main() {


    vec3 n = normalize(vnormal);
    float lambert = max(dot(n, lightdir), 0.0);
    float blinn = pow(max(dot(n, halfway), 0.0), 150.0);

    fragColor = vec4(
        rainbow(outY) * (lightcolor * lambert)
        +
        (lightcolor * blinn) * 1.0
    , color.a);
}