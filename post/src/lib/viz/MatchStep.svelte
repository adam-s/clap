<script lang="ts">
  /**
   * MatchStep — the payoff. Both the sound and every label are now points in one
   * 512-d space, so "does this sound match this phrase?" is just the angle between
   * two unit vectors: cosine = dot product. We compute it LIVE against the audio
   * embedding of whatever clip is selected, then softmax → a zero-shot label.
   */
  import type { ClapBundle, Clip } from '../data/clapData';

  type Props = { bundle: ClapBundle; clip: Clip; n: number };
  let { bundle, clip, n }: Props = $props();

  const items = $derived(bundle.text);
  const scale = $derived(bundle.meta.logit_scale ?? 10);

  // cosine = dot product of two L2-normalised vectors (audio is computed live)
  function dot(a: number[], b: number[]): number {
    let s = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }
  const scores = $derived(
    items.map((it) => ({ caption: it.caption, slug: it.slug, cos: dot(clip.audioEmbed, it.textEmbed) }))
  );
  const probs = $derived.by(() => {
    const z = scores.map((s) => s.cos * scale);
    if (!z.length) return [];                    // guard: empty label set → no softmax
    const m = Math.max(...z);
    const ex = z.map((v) => Math.exp(v - m));
    const sum = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / sum);
  });
  const winner = $derived(scores.reduce((best, s, i) => (s.cos > scores[best].cos ? i : best), 0));
  const fmt = (x: number) => (x >= 0 ? '+' : '') + x.toFixed(2);
  // map cosine [-1,1] → [0,1] bar fraction
  const frac = (c: number) => Math.max(0, Math.min(1, (c + 1) / 2));
</script>

<section class="step">
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · The match</span>
    <span class="dims mono">cosine · audio ↔ text</span>
  </header>

  <p class="explain">
    "Which words fit this sound?" is now nothing fancier than the angle between two
    unit vectors: <em>cosine similarity, a single dot product</em>. Higher means they
    point the same way. Run it against every label, softmax the scores, and you get a
    zero-shot guess — no classifier was ever trained on these categories.
  </p>

  <div class="verdict">
    <span class="vlab mono">heard:</span>
    <span class="vname">{scores[winner].caption}</span>
    <span class="vprob mono">{(probs[winner] * 100).toFixed(0)}%</span>
    {#if scores[winner].slug === clip.slug}
      <span class="vok mono">✓ matches the clip</span>
    {/if}
  </div>

  <div class="bars">
    {#each scores as s, i (s.slug)}
      <div class="bar" class:win={i === winner}>
        <span class="cap mono">{s.caption}</span>
        <div class="track">
          <span class="zero"></span>
          <span class="fill" style={`width:${frac(s.cos) * 100}%`}></span>
        </div>
        <span class="cos mono">{fmt(s.cos)}</span>
        <span class="pct mono">{(probs[i] * 100).toFixed(0)}%</span>
      </div>
    {/each}
  </div>

  <p class="foot">
    This is the whole point of the contrastive training: it pushed each sound and
    its caption to point the same way, and everything else apart. Swap the clip up
    top and watch the winner follow the sound.
  </p>
</section>

<style>
  .step { position: relative; display: flex; flex-direction: column; gap: 16px; padding: var(--space-xl) 0; border-top: 1px solid var(--line); }
  header { display: flex; align-items: baseline; gap: 14px; }
  .eyebrow { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .dims { color: var(--ink-faint); font-size: 0.72rem; margin-left: auto; }
  .explain, .foot { margin: 0; color: var(--ink-dim); font-size: 0.95rem; line-height: 1.6; max-width: 64ch; }
  .explain em { color: var(--ink); font-style: italic; }
  .foot { font-size: 0.88rem; color: var(--ink-faint); }

  .verdict { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
    padding: 12px 16px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
  .vlab { font-size: 0.74rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.06em; }
  .vname { font-size: 1.2rem; font-weight: 700; color: var(--data); }
  .vprob { font-size: 0.95rem; color: var(--ink-dim); }
  .vok { font-size: 0.72rem; color: #2e7d4f; margin-left: auto; }

  .bars { display: flex; flex-direction: column; gap: 10px; }
  .bar { display: grid; grid-template-columns: 150px 1fr 52px 44px; align-items: center; gap: 12px; }
  .cap { font-size: 0.82rem; color: var(--ink-dim); text-align: right; }
  .bar.win .cap { color: var(--ink); font-weight: 600; }
  .track { position: relative; height: 14px; background: var(--panel-2); border-radius: 3px; overflow: hidden; }
  .zero { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--line-strong); }
  .fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--data); opacity: 0.45;
    border-radius: 3px; transition: width 280ms ease; }
  .bar.win .fill { opacity: 1; }
  .cos { font-size: 0.78rem; color: var(--ink-dim); text-align: right; }
  .pct { font-size: 0.78rem; color: var(--ink-faint); text-align: right; }
  .bar.win .pct { color: var(--data); font-weight: 600; }

  @media (max-width: 560px) {
    .bar { grid-template-columns: 96px 1fr 44px 40px; gap: 8px; }
    .cap { font-size: 0.72rem; }
  }
</style>
