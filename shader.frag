
#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
uniform sampler2D image;

// Based on:
// Signed Distance to a Quadratic Bezier Curve
// - Adam Simmons (@adamjsimmons) 2015
// https://www.shadertoy.com/view/ltXSDB

vec3 solveCubic(float a, float b, float c)
{
    float p = b - a*a / 3.0, p3 = p*p*p;
    float q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
    float d = q*q + 4.0*p3 / 27.0;
    float offset = -a / 3.0;
    if(d >= 0.0) { 
        float z = sqrt(d);
        vec2 x = (vec2(z, -z) - q) / 2.0;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        return vec3(offset + uv.x + uv.y);
    }
    float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
    float m = cos(v), n = sin(v)*1.732050808;
    return vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) + offset;
}

vec3 sdBezier(vec2 A, vec2 B, vec2 C, vec2 p)
{
   B = mix(B + vec2(1e-4), B, abs(sign(B * 2.0 - A - C)));
   vec2 a = B - A, b = A - B * 2.0 + C, c = a * 2.0, d = A - p;
   vec3 k = vec3(3.*dot(a,b),2.*dot(a,a)+dot(d,b),dot(d,a)) / dot(b,b);

   vec2 t = clamp(solveCubic(k.x, k.y, k.z).xy, 0.0, 1.0);
   vec2 dp1 = d + (c + b*t.x)*t.x;
   float d1 = dot(dp1, dp1);
   vec2 dp2 = d + (c + b*t.y)*t.y;
   float d2 = dot(dp2, dp2);

   // note: 3rd root is actually never closest, we can just ignore it
   
   // Find closest distance and t
   vec2 r = (d1 < d2) ? vec2(d1, t.x) : vec2(d2, t.y);
   
   // Find on which side (t=0 or t=1) is extension
   vec2 e = vec2(step(0.,-r.y),step(1.,r.y));

   // Calc. gradient
   vec2 g = 2.*b*r.y + c;
   vec2 ng = normalize(g);
   
   // Calc. extension to t
   float et = (e.x*dot(-d,g) + e.y*dot(p-C,g))/dot(g,g);
   float net = (e.x*dot(-d,ng) + e.y*dot(p-C,ng))/dot(ng,ng); // it's for normalized t
   
   // Find closest point on curve with extension
   vec2 dp = d + (c + b*r.y)*r.y + et*g;
   
   // Sign is just cross product with gradient
   float s =  sign(g.x*dp.y - g.y*dp.x);
   
   return vec3(sqrt(r.x), s*length(dp), r.y + net);
}

void main()
{
    vec2 p = (2.0*gl_FragCoord.xy-u_resolution.xy)/u_resolution.xy;

    float time = sin(u_time*5.)* .25;
    vec2 m = (u_mouse.xy-u_resolution.xy)/u_resolution.xy + vec2(0.0, 1.0);

    // Define the control points of our curve
    vec2 A = vec2(0.0, -0.7);
    vec2 C = vec2(0.0, -0.45);
    vec2 CONTROL = m; // 控制点C

    // offset算法, 删去注释可变为自动震荡模式
    // CONTROL = A + vec2(sin(time), cos(time))*(C.y-A.y);
    // CONTROL = C + vec2(0., time* .3);
    
    vec2 B = vec2(0.0, A.y*.5 + CONTROL.y*.5);

    // Get the signed distance to bezier curve
    vec3 r = sdBezier(A, B, CONTROL, p);

    // 计算此dist和t下所对应的原始贴图坐标
    // 注意控制点的坐标系是 -1~1
    float ox = 0.5 + r.y* 0.5;
    // A -> C
    // -0.6 -> +0.5
    // (x+1.)/2. => 0.2 -> 0.75
    float oy;
    if (r.z > 1.){
        oy = (r.z + C.y)* .5;
    }
    else if (r.z < 0.){
        oy = (r.z + A.y + 1.)* .5;
    }
    else{
        oy = (r.z*(C.y - A.y) + A.y + 1.)* 0.5;
    }
    
    float d = r.x; // unsigned distance

    gl_FragColor = vec4(.7, .7, .7, 1.0);
    
    gl_FragColor = mix(texture2D(image, vec2(ox, oy)), gl_FragColor, 1.-texture2D(image, vec2(ox, oy)).a);

    // 调试: 贝塞尔曲线
    gl_FragColor = mix(gl_FragColor, vec4(0.0, 1.0, 0.0, 1.0), 1.0-smoothstep(0.01,0.015,abs(d)) );

    // 调试: 贝塞尔曲线控制点
    float pd = min(distance(p, A),(min(distance(p, B),distance(p, CONTROL))));
    gl_FragColor = mix(vec4(1.0 - smoothstep(0.02, 0.03, pd), vec2(0.), 1.), gl_FragColor, smoothstep(0.03, 0.04, pd));

    // 调试: 横向和纵向网格
    gl_FragColor.rgb -= smoothstep(.9,1.,cos(50.*(r.z)))*.1;
    gl_FragColor.rgb -= smoothstep(.9,1.,cos(50.*abs(r.y)))*.1;

    // 调试: 原始图像
    vec4 ori = texture2D(image, gl_FragCoord.xy/u_resolution.xy + vec2(.4, 0.));
    gl_FragColor.rgb = mix(ori.rgb, gl_FragColor.rgb, 1.0-0.2* ori.a);
}