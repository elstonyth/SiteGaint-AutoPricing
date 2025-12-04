const canvas = document.getElementById('bg-canvas');
const gl = canvas.getContext('webgl');

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener('resize', resize);
resize();

const vsSource = `
    attribute vec4 aVertexPosition;
    void main() {
        gl_Position = aVertexPosition;
    }
`;

const fsSource = `
    precision mediump float;
    uniform float uTime;
    uniform vec2 uResolution;

    void main() {
        vec2 uv = gl_FragCoord.xy / uResolution.xy;
        float time = uTime * 0.15; // Slow, majestic movement
        
        // Aspect ratio correction
        vec2 p = uv * 2.0 - 1.0;
        p.x *= uResolution.x / uResolution.y;

        // Neural Flow Simulation
        // Layer 1: Deep base movement
        float r = sin(p.x * 3.0 + time) + cos(p.y * 2.0 + time * 0.5);
        
        // Layer 2: Detail flow
        float g = sin(p.x * 6.0 - time * 0.8) + cos(p.y * 5.0 + time);
        
        // Layer 3: Interference
        float b = sin(length(p) * 4.0 - time * 1.2);
        
        // Composite colors
        // Deep midnight blue base
        vec3 color = vec3(0.02, 0.02, 0.05);
        
        // Add Electric Indigo nebulae
        float nebula = smoothstep(0.0, 2.0, r + g);
        color += vec3(0.2, 0.1, 0.5) * nebula * 0.4;
        
        // Add Cyan highlights (neural sparks)
        float spark = smoothstep(0.8, 1.0, sin(b * 3.0 + time));
        color += vec3(0.0, 0.8, 0.9) * spark * 0.15;

        // Vignette
        float vignette = 1.0 - length(uv - 0.5) * 0.5;
        color *= vignette;

        gl_FragColor = vec4(color, 1.0);
    }
`;

function initShaderProgram(gl, vsSource, fsSource) {
    const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);
    return shaderProgram;
}

function loadShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

const shaderProgram = initShaderProgram(gl, vsSource, fsSource);

const programInfo = {
    program: shaderProgram,
    attribLocations: {
        vertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
    },
    uniformLocations: {
        resolution: gl.getUniformLocation(shaderProgram, 'uResolution'),
        time: gl.getUniformLocation(shaderProgram, 'uTime'),
    },
};

const positionBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
const positions = [
    -1.0,  1.0,
     1.0,  1.0,
    -1.0, -1.0,
     1.0, -1.0,
];
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

function render(now) {
    now *= 0.001; // convert to seconds

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(programInfo.program);

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(programInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);

    gl.uniform2f(programInfo.uniformLocations.resolution, canvas.width, canvas.height);
    gl.uniform1f(programInfo.uniformLocations.time, now);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    requestAnimationFrame(render);
}

requestAnimationFrame(render);
