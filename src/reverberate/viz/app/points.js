/** The measurement points of the fields, as small dots in the room, shown
 * on request. A dot fades out as the listener comes within arm's reach of
 * it, so standing on a point never fills the view with the point.
 */
import * as THREE from "three";

const NEAR_M = 0.35;
const FAR_M = 0.9;

export function createPoints(viewport) {
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: {
      eye: { value: new THREE.Vector3() },
      colour: { value: new THREE.Color(0xf2a541) },
      pixelRatio: { value: Math.min(devicePixelRatio, 2) },
      near: { value: NEAR_M },
      far: { value: FAR_M },
    },
    vertexShader: `
      uniform vec3 eye; uniform float pixelRatio; uniform float near; uniform float far;
      varying float vAlpha;
      void main() {
        float d = distance(position, eye);
        vAlpha = smoothstep(near, far, d);
        vec4 view = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = clamp(220.0 * pixelRatio / max(-view.z, 0.2), 3.0, 40.0) * 0.06;
        gl_Position = projectionMatrix * view;
      }`,
    fragmentShader: `
      uniform vec3 colour; varying float vAlpha;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float r = length(c);
        if (r > 0.5) discard;
        float edge = smoothstep(0.5, 0.35, r);
        gl_FragColor = vec4(colour, edge * vAlpha * 0.9);
      }`,
  });
  const geometry = new THREE.BufferGeometry();
  const points = new THREE.Points(geometry, material);
  points.name = "measurement-points";
  points.visible = false;
  points.frustumCulled = false;
  viewport.overlays.add(points);
  let dots = [];

  return {
    /** The union of every field's cell centres, once per distinct position. */
    set(fields) {
      const seen = new Map();
      for (const field of fields) {
        for (const [x, y, z] of field.positions()) {
          seen.set(`${x.toFixed(3)},${y.toFixed(3)},${z.toFixed(3)}`, [x, y, z]);
        }
      }
      dots = [...seen.values()];
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(dots.flat(), 3));
      geometry.computeBoundingSphere();
    },
    positions: () => dots,
    show(on) {
      points.visible = on;
    },
    /** Called every frame with the listener's position. */
    follow(x, y, z) {
      material.uniforms.eye.value.set(x, y, z);
    },
  };
}
