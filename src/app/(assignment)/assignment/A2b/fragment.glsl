/////////////////////////////////////////////////////
//// CS 8803/4803 CGAI: Computer Graphics in AI Era
//// Assignment 2B: Neural SDF
/////////////////////////////////////////////////////

precision highp float;              //// set default precision of float variables to high precision

varying vec2 vUv;                   //// screen uv coordinates (varying, from vertex shader)
uniform vec2 iResolution;           //// screen resolution (uniform, from CPU)
uniform float iTime;                //// time elapsed (uniform, from CPU)

#define PI 3.1415925359

const vec3 CAM_POS = vec3(0, 1, 0);

vec3 rotate(vec3 p, vec3 ax, float ro)
{
    return mix(dot(p, ax) * ax, p, cos(ro)) + sin(ro) * cross(ax, p);
}

/////////////////////////////////////////////////////
//// sdf functions
/////////////////////////////////////////////////////

float sdfPlane(vec3 p, float h)
{
    return p.y - h;
}

float sdfBunny(vec3 p)
{
    p = rotate(p, vec3(1., 0., 0.), PI / 2.);
    p = rotate(p, vec3(0., 0., 1.), PI / 2. + PI / 1.);

    // sdf is undefined outside the unit sphere, uncomment to witness the abominations
    if(length(p) > 1.0)
    {
        return length(p) - 0.9;
    }

    //// neural network weights for the bunny 

    vec4 f0_0=sin(p.y*vec4(1.74,-2.67,1.91,-1.93)+p.z*vec4(2.15,-3.05,.50,-1.32)+p.x*vec4(2.47,.30,-2.00,-2.75)+vec4(1.31,6.89,-8.25,.15));
    vec4 f0_1=sin(p.y*vec4(-.72,-3.13,4.36,-3.50)+p.z*vec4(3.39,3.58,-4.52,-1.10)+p.x*vec4(-1.02,-2.90,2.23,-.62)+vec4(1.61,-.84,-2.00,-.47));
    vec4 f0_2=sin(p.y*vec4(-1.47,.32,-.70,-1.51)+p.z*vec4(.17,.75,3.59,4.05)+p.x*vec4(-3.10,1.40,4.72,2.90)+vec4(-6.76,-6.43,2.41,-.66));
    vec4 f0_3=sin(p.y*vec4(-2.75,1.59,3.43,-3.39)+p.z*vec4(4.09,4.09,-2.34,1.23)+p.x*vec4(1.07,.65,-.18,-3.46)+vec4(-5.09,.73,3.06,3.35));
    vec4 f1_0=sin(mat4(.47,.12,-.23,-.04,.48,.06,-.24,.19,.12,.72,-.08,.39,.37,-.14,-.01,.06)*f0_0+
        mat4(-.62,-.40,-.81,-.30,-.34,.08,.26,.37,-.16,.38,-.09,.36,.02,-.50,.34,-.38)*f0_1+
        mat4(-.26,-.51,-.32,.32,-.67,.35,-.43,.93,.12,.34,.07,-.01,.67,.27,.43,-.02)*f0_2+
        mat4(.02,-.18,-.15,-.10,.47,-.07,.82,-.46,.18,.44,.39,-.94,-.20,-.28,-.20,.29)*f0_3+
        vec4(-.09,-3.49,2.17,-1.45))/1.0+f0_0;
    vec4 f1_1=sin(mat4(-.46,-.33,-.85,-.57,.41,.87,.25,.58,-.47,.16,-.14,-.06,-.70,-.82,-.20,.47)*f0_0+
        mat4(-.15,-.73,-.46,-.58,-.54,-.34,-.02,.12,.55,.32,.22,-.87,-.57,-.28,-.51,.10)*f0_1+
        mat4(.75,1.06,-.08,-.17,-.43,.69,1.07,.23,.46,-.02,.10,-.11,.21,-.70,-.08,-.48)*f0_2+
        mat4(.04,-.09,-.51,-.06,1.12,-.21,-.35,-.17,-.95,.49,.22,.99,.62,-.25,.06,-.20)*f0_3+
        vec4(-.61,2.91,-.17,.71))/1.0+f0_1;
    vec4 f1_2=sin(mat4(.01,-.86,-.07,.46,.73,-.28,.83,.12,.16,.33,.28,-.55,-.21,-.02,.53,-.15)*f0_0+
        mat4(-.28,-.32,.19,-.28,.24,-.23,-.61,-.39,.26,.40,.18,.41,.21,.57,-.91,-.29)*f0_1+
        mat4(.23,-.40,-1.34,-.50,.08,-.04,-1.67,-.16,-.65,-.09,.38,-.22,-.14,-.34,.37,.05)*f0_2+
        mat4(-.47,-.23,-.57,-.05,.51,.04,.00,.27,.80,.29,-.09,-.53,-.20,-.41,-.64,-.12)*f0_3+
        vec4(1.08,4.00,-2.54,2.18))/1.0+f0_2;
    vec4 f1_3=sin(mat4(-.30,.38,.39,.53,.73,.73,-.06,.01,.54,-.07,-.19,.68,.59,.40,.04,.07)*f0_0+
        mat4(-.17,.44,-.61,.43,-.84,-.12,.65,-.50,.33,-.31,-.28,.13,.18,-.42,.14,.08)*f0_1+
        mat4(-.78,.06,-.18,.37,-.99,.49,.71,.15,.27,-.48,-.17,.25,.05,.10,-.40,-.21)*f0_2+
        mat4(-.17,-.27,.40,.18,-.24,.23,.03,-.83,-.30,-.38,.07,.21,-.45,-.24,.78,.50)*f0_3+
        vec4(2.14,-3.48,3.81,-1.43))/1.0+f0_3;
    vec4 f2_0=sin(mat4(.83,.15,-.49,-.80,-.83,.16,1.24,.75,-.27,.18,-.13,1.05,.70,-.15,.30,.79)*f1_0+
        mat4(-.38,-.17,.34,.67,-.39,.09,.48,-.93,.19,.60,-.20,-.22,-.76,-.62,-.40,.01)*f1_1+
        mat4(.10,.22,.08,.13,-.42,-.11,.71,-.63,.02,.46,-.07,-.46,-.37,.07,.15,.14)*f1_2+
        mat4(.09,-.48,-.38,.40,-.57,-.88,-.14,-.25,.20,.95,.86,-1.08,.46,.04,.53,-.82)*f1_3+
        vec4(3.47,-3.66,3.06,.84))/1.4+f1_0;
    vec4 f2_1=sin(mat4(1.03,.03,-.76,-.03,.84,.66,-.49,.74,-.09,-.85,-.55,.17,.07,.85,-.55,-.20)*f1_0+
        mat4(-.55,1.13,.41,-.21,-.55,.19,.49,.67,.40,1.80,-.82,-.83,-1.02,.78,-.42,-.51)*f1_1+
        mat4(.77,-.88,.64,1.10,-.49,1.05,-.43,-.38,.66,-.63,.02,.11,-.24,-.23,.49,-.65)*f1_2+
        mat4(-.66,1.90,.02,-.48,.22,-.62,-.68,-.44,.52,-.57,.16,-.61,-.03,-.02,-.88,-.23)*f1_3+
        vec4(.58,-3.00,-2.53,.14))/1.4+f1_1;
    vec4 f2_2=sin(mat4(-.44,-.06,.30,-.37,.27,-.23,-.56,.15,.03,-.14,-.08,.72,.76,-.58,.55,.29)*f1_0+
        mat4(.31,.23,.42,-.17,.37,-.05,.39,.46,-1.14,.32,.06,-.28,.28,-.21,-.58,.62)*f1_1+
        mat4(.92,-.16,.86,-.09,-.12,.33,-.49,-.24,.29,-.19,.95,-.40,-.87,.08,.08,-.71)*f1_2+
        mat4(-.45,.67,1.07,-.14,-.56,.06,-.81,-.15,-.57,-.24,-1.09,.69,-.44,-.32,-.00,-.07)*f1_3+
        vec4(-4.43,-1.86,-2.87,1.45))/1.4+f1_2;
    vec4 f2_3=sin(mat4(.58,.25,.01,-.54,.34,.56,.61,-.79,-.01,.05,-.57,-1.31,.74,.78,-.10,-.11)*f1_0+
        mat4(-.03,-.48,-.24,.01,.10,.23,.22,-.05,.76,.29,-.37,.02,.54,-.07,.27,.38)*f1_1+
        mat4(.31,-1.03,.24,.95,.80,.29,.43,.61,-.04,-.22,-.06,-.52,-.46,.35,.07,-.07)*f1_2+
        mat4(.47,-.12,-.62,.06,.47,-.41,.53,-2.14,-.59,.16,.74,-.58,.32,.66,-.30,-.18)*f1_3+
        vec4(-2.86,-3.27,-.55,2.87))/1.4+f1_3;
    return dot(f2_0,vec4(-.08,.03,.07,-.03))+
        dot(f2_1,vec4(-.03,-.02,-.06,-.07))+
        dot(f2_2,vec4(.05,-.09,.03,.11))+
        dot(f2_3,vec4(.03,.06,-.06,-.03))+
        -0.014;
}

/////////////////////////////////////////////////////
//// Step 1: training a neural SDF model
//// You are asked to train your own neural SDF model on Colab. 
//// Your implementation should take place in neural_sdf.ipynb.
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 2: copy neural SDF weights to GLSL
//// In this step, you are asked to the network weights you have trained from the text file to the function sdfCow().
//// You should replace the default implementation (a sphere) with your own network weights. 
/////////////////////////////////////////////////////

float sdfCow(vec3 p)
{
    p = rotate(p, vec3(1., 0., 0.), PI / 2.);
    p = rotate(p, vec3(0., 0., 1.), PI / 3. + PI/3.0);

    // sdf is undefined outside the unit sphere, uncomment to witness the abominations
    if(length(p) > 1.)
    {
        return length(p) - 0.9;
    }

    //// your implementation starts
    vec4 f0_0=sin(p.y*vec4(-2.67,-1.16,1.81,2.79)+p.z*vec4(-2.83,1.71,-3.03,-1.46)+p.x*vec4(-.92,-4.19,.79,-1.57)+vec4(4.39,4.62,-6.34,-8.10));
    vec4 f0_1=sin(p.y*vec4(-.54,-2.94,2.28,3.80)+p.z*vec4(-1.68,-4.43,-2.17,-2.17)+p.x*vec4(1.56,1.18,3.83,-4.28)+vec4(-2.81,7.89,-1.08,-3.44));
    vec4 f0_2=sin(p.y*vec4(-2.47,.69,-1.39,-2.31)+p.z*vec4(.77,2.51,4.45,.65)+p.x*vec4(2.61,2.24,-2.11,-1.78)+vec4(-1.88,-2.35,3.35,2.31));
    vec4 f0_3=sin(p.y*vec4(3.00,.84,2.63,-3.57)+p.z*vec4(1.50,-2.85,-1.12,-.47)+p.x*vec4(-2.56,-.49,2.51,1.45)+vec4(1.81,1.54,1.93,1.46));
    vec4 f1_0=sin(mat4(.03,.80,-.09,.15,.08,-.09,-.21,.58,-.31,.47,-.42,-.52,.10,-.25,.09,-.17)*f0_0+
        mat4(-.37,.86,.18,-.35,.24,-.17,-.07,.46,.55,.41,-.09,-.66,-.18,-.39,-.18,.31)*f0_1+
        mat4(.78,-.51,-.53,-.19,.15,.05,.20,-.48,.25,.30,.27,.75,-.30,.01,-.76,-.04)*f0_2+
        mat4(-.35,-.47,.06,-.81,.14,-.27,.50,-.55,.02,-.23,.54,.23,.40,-.19,-.12,.54)*f0_3+
        vec4(2.72,3.20,-.86,-1.81))/1.0+f0_0;
    vec4 f1_1=sin(mat4(-.78,.34,-.51,.28,-.09,.03,.24,.05,-.12,-.92,.28,.48,.04,-.07,.05,.61)*f0_0+
        mat4(.16,.85,-.01,-.20,.05,.55,-.14,-.55,.34,-.61,-.20,.00,.05,-.28,-.70,.06)*f0_1+
        mat4(-.31,.37,.60,-.14,.54,.41,.11,.37,.08,.27,-.66,-.27,.05,-.90,-.43,.41)*f0_2+
        mat4(-.42,-.23,.35,-.45,.73,.12,-.02,.01,.41,-.40,.70,.19,.13,-.70,.20,.20)*f0_3+
        vec4(-1.36,2.88,.87,3.40))/1.0+f0_1;
    vec4 f1_2=sin(mat4(.37,.33,-.20,.06,.29,-.61,-.08,.26,-.58,.51,-.64,-.58,-.61,.01,.14,.22)*f0_0+
        mat4(.42,.61,.29,.19,.48,.28,-.26,-.10,.13,.16,.19,-.12,.75,-.22,-.07,.24)*f0_1+
        mat4(.29,-.26,-.06,.97,-.21,.22,-.05,-.30,.53,-.32,.57,.13,1.21,-.48,-.73,-.19)*f0_2+
        mat4(.67,.43,.45,.27,-.05,-.14,.61,-.12,-.14,.24,.21,.43,.44,-.05,-.40,.16)*f0_3+
        vec4(1.50,.87,1.94,3.52))/1.0+f0_2;
    vec4 f1_3=sin(mat4(-.55,.16,-.46,.30,-.18,-.58,-.04,-.51,-.13,-.56,.05,-.17,.87,-.35,-.41,.66)*f0_0+
        mat4(.83,.36,-.12,-.03,-.50,.20,.28,.15,.48,.11,.09,.33,.11,-.55,-.16,.30)*f0_1+
        mat4(-.60,-.30,-.15,.17,.11,-.27,.26,-.04,-.46,.70,-.63,-.17,-.89,.54,.54,.10)*f0_2+
        mat4(-.43,.23,-.15,.45,-.62,-.50,.53,.33,.38,-.39,.25,.03,.59,.26,-.16,.32)*f0_3+
        vec4(2.99,-3.42,-.08,-2.22))/1.0+f0_3;
    vec4 f2_0=sin(mat4(-.59,-.10,-.14,-.42,-.59,-.18,.44,-.27,.41,.52,-.46,-.65,-.35,-.27,-.14,.57)*f1_0+
        mat4(.11,-.66,.39,-.40,-.34,-.50,-.26,.41,-.58,.52,.19,.63,-.17,.59,.25,.32)*f1_1+
        mat4(-.00,.72,.29,-.35,-.67,.55,.01,.63,-.43,-.54,.50,.17,-.40,.12,.07,.13)*f1_2+
        mat4(.06,-.01,-.34,.58,.18,-.12,.02,.29,-.10,-.20,-.30,-.09,.34,-.24,.75,-.17)*f1_3+
        vec4(1.16,4.03,-1.95,3.50))/1.4+f1_0;
    vec4 f2_1=sin(mat4(-.29,.39,-.41,.56,-.17,-.30,.45,-.19,-.31,-.00,-.60,-.02,-.05,-1.00,.84,-.25)*f1_0+
        mat4(.32,.06,.78,.11,-.05,.34,.15,.01,.06,.39,.05,-.38,-.02,-.13,.26,-.33)*f1_1+
        mat4(-.13,-.05,-.19,-.16,.65,-.05,-.72,.36,.03,.53,-.01,-.33,.35,.31,.20,-.94)*f1_2+
        mat4(.46,-.22,.92,-.41,.21,.20,.57,.40,-.53,.13,-.64,.27,-.52,-.85,.20,.35)*f1_3+
        vec4(3.68,3.93,-2.42,-2.38))/1.4+f1_1;
    vec4 f2_2=sin(mat4(.26,-.30,.22,.99,.14,.31,-.56,.85,-1.28,.40,.47,.02,1.05,-.15,-.44,.01)*f1_0+
        mat4(.12,-.24,.22,.10,.12,-.53,.02,.09,-.56,.00,.42,-.22,-.30,.09,-.12,.02)*f1_1+
        mat4(-.72,.34,-.07,-.23,1.18,.27,.05,-.17,-.14,.27,-.11,-.49,.54,.24,-.63,-.22)*f1_2+
        mat4(.44,.06,-.13,-.80,-.66,.02,.32,-.35,.67,.59,.41,-.40,-.37,.65,-.23,-.01)*f1_3+
        vec4(1.97,-3.73,1.88,-.87))/1.4+f1_2;
    vec4 f2_3=sin(mat4(.17,-.43,-.29,.13,-.02,.05,.33,.04,-.19,.11,.80,-1.06,.24,.38,-.56,.28)*f1_0+
        mat4(-.87,-.02,.59,-.39,.21,.40,-.88,.18,-.25,-.67,-.39,-.49,-.04,-.17,.05,-.07)*f1_1+
        mat4(.30,-.03,.25,.04,.28,.01,.38,-.29,.25,-.69,-.34,-.89,.19,-.46,.04,.54)*f1_2+
        mat4(-.16,-.23,.86,.79,.17,.30,-.29,.25,.06,-.12,.75,-.09,.25,.67,-1.45,.40)*f1_3+
        vec4(-2.62,-3.14,-1.68,2.84))/1.4+f1_3;
    return dot(f2_0,vec4(-.04,-.06,-.05,-.02))+
        dot(f2_1,vec4(-.04,-.04,-.02,-.05))+
        dot(f2_2,vec4(-.01,.07,.04,-.03))+
        dot(f2_3,vec4(-.09,.05,-.03,-.04))+
        -0.166;
    //// your implementation ends
}

float sdfDuck(vec3 p)
{
    p = rotate(p, vec3(1., 0., 0.), PI / 2.);
    p = rotate(p, vec3(0., 0., 1.), PI / 3. + PI/3.0);

    // sdf is undefined outside the unit sphere, uncomment to witness the abominations
    if(length(p) > 1.)
    {
        return length(p) - 0.9;
    }

    // PASTE 
}

float sdfUnion(float d1, float d2)
{
    return min(d1, d2);
}

/////////////////////////////////////////////////////
//// Step 3: scene sdf
//// You are asked to use the sdf boolean operations to draw the bunny and the cow in the scene.
//// The bunny is located in the ceter of vec3(-1.0, 1., 4.), and the cow is located in the center of vec3(1.0, 1., 4.).
/////////////////////////////////////////////////////

//// sdf: p - query point
float sdf(vec3 p)
{
    float s = 0.;

    float plane_h = -0.1;

    //// calculate the sdf based on all objects in the scene

    //// your implementation starts
    float bunny = sdfBunny(p - vec3(-1.0, 1.0, 4.0));
    float cow   = sdfCow(p - vec3(1.0, 1.0, 4.0));
    float plane = sdfPlane(p, plane_h);

    s = sdfUnion(sdfUnion(bunny, cow), plane);
    //// your implementation ends

    return s;
}

/////////////////////////////////////////////////////
//// ray marching
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 4: ray marching
//// You are asked to implement the ray marching algorithm within the following for-loop.
//// You are allowed to reuse your previous implementation in A2a for this function.
/////////////////////////////////////////////////////

//// ray marching: origin - ray origin; dir - ray direction 
float rayMarching(vec3 origin, vec3 dir)
{
    float s = 0.0;

    //// your implementation starts
    for (int i = 0; i < 128; i++)
    {
        vec3 p = origin + s * dir;
        float d = sdf(p);
        if (d < 0.001) break;
        s += d;
        if (s > 100.0) break;
    }
    //// your implementation ends

    return s;
}

/////////////////////////////////////////////////////
//// normal calculation
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 5: normal calculation
//// You are asked to calculate the sdf normal based on finite difference.
//// You are allowed to reuse your previous implementation in A2a for this function.
/////////////////////////////////////////////////////

//// normal: p - query point
vec3 normal(vec3 p)
{
    float s = sdf(p);          //// sdf value in p
    float dx = 0.01;           //// step size for finite difference

    //// your implementation starts
    float da = sdf(p + vec3(dx, 0, 0)) - sdf(p - vec3(dx, 0, 0));
    float db = sdf(p + vec3(0, dx, 0)) - sdf(p - vec3(0, dx, 0));
    float dc = sdf(p + vec3(0, 0, dx)) - sdf(p - vec3(0, 0, dx));
    vec3 n = normalize(vec3(da, db, dc));
    //// your implementation ends

    return n;
}

/////////////////////////////////////////////////////
//// Phong shading
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 6: lighting and coloring
//// You are asked to specify the color for the two neural SDF objects in the scene.
//// Each object must have a separate color without mixing.
//// Notice that we have implemented the default Phong shading model for you.
/////////////////////////////////////////////////////

vec3 phong_shading(vec3 p, vec3 n)
{
    //// background
    if(p.z > 20.0)
    {
        vec3 color = vec3(0.04, 0.16, 0.33);
        return color;
    }

    //// phong shading
    vec3 lightPos = vec3(4. * sin(iTime), 4., 4. * cos(iTime));
    vec3 l = normalize(lightPos - p);
    float amb = 0.1;
    float dif = max(dot(n, l), 0.) * 0.7;
    vec3 eye = CAM_POS;
    float spec = pow(max(dot(reflect(-l, n), normalize(eye - p)), 0.0), 128.0) * 0.9;

    vec3 sunDir = normalize(vec3(0, 1, -1)); //// parallel light direction
    float sunDif = max(dot(n, sunDir), 0.) * 0.2;

    //// shadow
    float s = rayMarching(p + n * 0.02, l);
    if(s < length(lightPos - p))
        dif *= .2;

    vec3 color = vec3(1.0);

    //// your implementation starts
    float dBunny = sdfBunny(p - vec3(-1.0, 1.0, 4.0));
    float dCow   = sdfCow(p - vec3(1.0, 1.0, 4.0));
    float dPlane = sdfPlane(p, -0.1);

    if (dPlane < dBunny && dPlane < dCow)
        color = vec3(0.8);                      // ground plane
    else if (dBunny < dCow)
        color = vec3(0.95, 0.75, 0.55);         // bunny
    else
        color = vec3(0.90, 0.90, 0.95);         // cow

    //// your implementation ends

    return (amb + dif + spec + sunDif) * color;
}

/////////////////////////////////////////////////////
//// main function
/////////////////////////////////////////////////////

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord.xy - .5 * iResolution.xy) / iResolution.y;         //// screen uv
    vec3 origin = CAM_POS;                                                  //// camera position 
    vec3 dir = normalize(vec3(uv.x, uv.y, 1));                              //// camera direction
    float s = rayMarching(origin, dir);                                     //// ray marching
    vec3 p = origin + dir * s;                                              //// ray-sdf intersection
    vec3 n = normal(p);                                                     //// sdf normal
    vec3 color = phong_shading(p, n);                                       //// phong shading
    fragColor = vec4(color, 1.);                                            //// fragment color
}

void main()
{
    mainImage(gl_FragColor, gl_FragCoord.xy);
}