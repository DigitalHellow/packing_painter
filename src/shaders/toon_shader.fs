#version 330

/* Main function, uniforms & utils */
#ifdef GL_ES
    precision mediump float;
#endif

in vec2 TexCoord;

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
uniform sampler2D u_tex0; //../../imgs/flower.jpeg
uniform bool u_change_colors;

#define PI_TWO 1.570796326794897
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586

// All components are in the range [0…1], including hue.
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}


// All components are in the range [0…1], including hue.
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}


vec3 hue2rgb(float H)
{
    float R = abs(H * 6. - 3.) - 1.;
    float G = 2. - abs(H * 6. - 2.);
    float B = 2. - abs(H * 6. - 4.);
    return vec3 (clamp(R, 0.0, 1.0),clamp(G, 0.0, 1.0), clamp(B, 0.0, 1.0));
}


vec3 rgb2hsl(vec3 RGB)
{
    vec3 hsv = rgb2hsv(RGB);
    float L = hsv.z - hsv.y * 0.5;
    float S = hsv.y / (1. - abs(L * 2. - 1.) + 0.001);
    return vec3(hsv.x, S, L);
}


vec3 hsl2rgb(vec3 HSL)
{
    vec3 rgb = hue2rgb(HSL.x);
    float C = (1. - abs(2. * HSL.z - 1.)) * HSL.y;
    return (rgb - 0.5) * C + HSL.z;
}


float roundoff(float value, float prec)
{
    float pow_10 = pow(10.0, prec);
    return round(value * pow_10) / pow_10;
}


float rand1d(float v)
{
    return cos(v + cos(v * 90.1415) * 100.1415) * 0.5 + 0.5;
}


void main() 
{
    vec4 color2;
    
    vec4 image = texture(u_tex0, TexCoord);
    
    if (u_change_colors)
    {
        image.r += 0.3 * sin(u_time) + 0.2;
        image.g += 0.4 * sin(u_time / 2.0) + 0.1;
        image.b += 0.3 * cos(u_time / 4.0) + 0.2;
    }

    image.r = roundoff(image.r, .6);
    image.g = roundoff(image.g, .6);
    image.b = roundoff(image.b, .6);

    //image.rgb = mix(image.rgb, vec3(0.6549, 0.2275, 0.7882), 0.2);

    gl_FragColor = image;
}