<script lang="ts">
  /**
   * HeroSphere — the opening. 2,600 REAL LAION-CLAP embeddings of real car-fault
   * sounds (unit 512-d → they live on a sphere), projected to 3-D and coloured by
   * k-means group. A handful of named faults are placed at their centroid and
   * tethered by a spline. Dark ink on the transparent page, slow tumble.
   *
   * Perf: positions live in typed arrays; each frame rotates in place, sorts an
   * index array (no allocation), and draws — and skips all work when off-screen.
   */
  import { onMount, onDestroy } from 'svelte';

  type Lab = { label: string; xyz: [number, number, number]; n: number };
  let canvas: HTMLCanvasElement;
  let wrap: HTMLDivElement;
  let raf = 0, lastTs = 0, spin = 0, visible = true;

  const PAL = ['#b8551f', '#e0912f', '#8a4b6b', '#2f7d78', '#3b6ea5', '#9c6b2f', '#c25e4a'];

  // typed buffers, allocated once when data arrives
  let n = 0;
  let pos: Float32Array;          // n*3 unit-sphere coords
  let grp: Uint8Array;            // n group index
  let labels: Lab[] = [];
  const NB = 8;   // depth buckets — batch draws so alpha/colour set ~56×, not 2,600×

  function load(d: { points?: any[]; labels?: Lab[] }) {
    const pts = Array.isArray(d?.points) ? d.points : [];   // guard bad/old schema
    n = pts.length;
    pos = new Float32Array(n * 3); grp = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      pos[i * 3] = p.xyz[0]; pos[i * 3 + 1] = p.xyz[1]; pos[i * 3 + 2] = p.xyz[2];
      grp[i] = p.g | 0;
    }
    labels = Array.isArray(d?.labels) ? d.labels : [];
  }

  function frame(ts: number) {
    raf = requestAnimationFrame(frame);
    const dt = lastTs ? (ts - lastTs) / 1000 : 0;
    lastTs = ts;
    if (!visible || !n) return;                          // off-screen → no work
    spin += dt;
    const yaw = spin * 0.14, pitch = 0.32 + Math.sin(spin * 0.16) * 0.26;
    const cyaw = Math.cos(yaw), syaw = Math.sin(yaw), cpit = Math.cos(pitch), spit = Math.sin(pitch);

    const ctx = canvas.getContext('2d')!;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (!W || !H) return;                                // not laid out yet
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr; canvas.height = H * dpr;   // fix: track height too
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.32, cam = 3.6;
    const G = PAL.length;

    // Bucket points by depth × group into Path2Ds, then fill each bucket once.
    // Depth order is preserved at bucket granularity; ~NB×G fills instead of n.
    const paths: Path2D[] = new Array(NB * G);
    for (let j = 0; j < paths.length; j++) paths[j] = new Path2D();
    for (let i = 0; i < n; i++) {
      const x0 = pos[i * 3], y0 = pos[i * 3 + 1], z0 = pos[i * 3 + 2];
      const x = x0 * cyaw + z0 * syaw;
      let z = -x0 * syaw + z0 * cyaw;
      const y = y0 * cpit - z * spit;
      z = y0 * spit + z * cpit;
      const s = cam / (cam - z);
      const px = cx + x * R * s, py = cy - y * R * s;
      let b = ((z + 1) / 2 * NB) | 0; if (b < 0) b = 0; else if (b >= NB) b = NB - 1;
      const r = 0.8 + ((b + 0.5) / NB) * 2.1;
      const p = paths[b * G + (grp[i] % G)];
      p.moveTo(px + r, py); p.arc(px, py, r, 0, 6.283);
    }
    for (let b = 0; b < NB; b++) {
      ctx.globalAlpha = 0.1 + ((b + 0.5) / NB) * 0.82;
      for (let g = 0; g < G; g++) { ctx.fillStyle = PAL[g]; ctx.fill(paths[b * G + g]); }
    }
    ctx.globalAlpha = 1;

    // named-fault landmarks (leader + side label) — hidden on narrow canvases
    if (W >= 520) {
      ctx.font = '600 12.5px "JetBrains Mono", monospace';
      ctx.lineWidth = 1;
      for (const L of labels) {
        const x0 = L.xyz[0], y0 = L.xyz[1], z0 = L.xyz[2];
        const x = x0 * cyaw + z0 * syaw;
        let z = -x0 * syaw + z0 * cyaw;
        const y = y0 * cpit - z * spit;
        z = y0 * spit + z * cpit;
        const s = cam / (cam - z);
        const px = cx + x * R * s, py = cy - y * R * s;
        const front = (z + 1) / 2;
        const a = Math.max(0, (front - 0.42) * 1.9);
        if (a < 0.03) continue;
        const left = px < cx;
        let ax = left ? cx - (R * 1.4 + 20) : cx + (R * 1.4 + 20);
        ax = Math.max(64, Math.min(W - 64, ax));         // keep the label on-canvas
        const ay = Math.max(26, Math.min(H - 26, py));
        ctx.globalAlpha = Math.min(1, a) * 0.5;
        ctx.strokeStyle = '#5a4a44';
        ctx.beginPath(); ctx.moveTo(px, py);
        ctx.quadraticCurveTo((px + ax) / 2 + (left ? -28 : 28), (py + ay) / 2, ax, ay); ctx.stroke();
        ctx.globalAlpha = Math.min(1, a);
        ctx.fillStyle = '#2a2024';
        ctx.beginPath(); ctx.arc(px, py, 3, 0, 6.283); ctx.fill();
        ctx.beginPath(); ctx.arc(px, py, 5.5, 0, 6.283); ctx.stroke();
        ctx.textAlign = left ? 'right' : 'left';
        ctx.fillText(L.label, ax + (left ? -8 : 8), ay + 4);
      }
      ctx.textAlign = 'left'; ctx.globalAlpha = 1;
    }
  }

  onMount(() => {
    fetch(`${import.meta.env.BASE_URL}data/hero.json`).then((r) => r.json()).then(load);
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    if (wrap) io.observe(wrap);
    raf = requestAnimationFrame(frame);
    return () => io.disconnect();
  });
  onDestroy(() => cancelAnimationFrame(raf));
</script>

<div class="hero-sphere" bind:this={wrap} role="img"
  aria-label="A rotating sphere of 2,600 real CLAP embeddings of car sounds, grouped by colour, with a few named faults labelled.">
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .hero-sphere {
    position: relative;
    width: min(920px, 94vw);
    margin-left: 50%;
    transform: translateX(-50%);
    height: clamp(400px, 52vw, 560px);
  }
  canvas { display: block; width: 100%; height: 100%; }
</style>
