<script lang="ts">
  import { onDestroy } from 'svelte';
  import { loadClap, type ClapBundle, type Clip } from './lib/data/clapData';
  import { activeClip, audioUrl, initLive, selectDemo } from './lib/data/clapLive.svelte';
  import { Transport } from './lib/audio/transport.svelte';
  import Layout from './lib/components/Layout.svelte';
  import Section from './lib/components/Section.svelte';
  import Prose from './lib/components/Prose.svelte';
  import ClipPicker from './lib/components/ClipPicker.svelte';
  import SpectrogramStep from './lib/viz/SpectrogramStep.svelte';
  import ConvStep from './lib/viz/ConvStep.svelte';
  import HierarchyStep from './lib/viz/HierarchyStep.svelte';
  import PoolStep from './lib/viz/PoolStep.svelte';
  import ProjectionStep from './lib/viz/ProjectionStep.svelte';
  import TextPicker from './lib/components/TextPicker.svelte';
  import TokenLookupStep from './lib/viz/TokenLookupStep.svelte';
  import ContextPoolStep from './lib/viz/ContextPoolStep.svelte';
  import TextProjectStep from './lib/viz/TextProjectStep.svelte';
  import MatchStep from './lib/viz/MatchStep.svelte';
  import MatrixStep from './lib/viz/MatrixStep.svelte';
  import HeroSphere from './lib/viz/HeroSphere.svelte';
  import FooterBar from './lib/components/FooterBar.svelte';

  let bundle = $state<ClapBundle | null>(null);
  let textSel = $state(0);
  let err = $state<string | null>(null);
  loadClap()
    .then((b) => {
      bundle = b;
      initLive(b.weights, b.clips, b.meta.win_sec ?? 2.0);
      selectDemo(0);
    })
    .catch((e) => (err = String(e)));

  let active = $state<Clip | null>(null);
  const u = activeClip.subscribe((v) => (active = v));

  // One persistent transport for the whole page; a clip change just re-points it,
  // so the footer and step 01 share one clock and nothing remounts.
  const transport = new Transport();
  const uAudio = audioUrl.subscribe((url) => { if (url) transport.load(url); });
  onDestroy(() => { u(); uAudio(); transport.destroy(); });

</script>

{#if err}
  <div class="boot err">Failed to load: {err}</div>
{:else if !bundle}
  <div class="boot">Loading…</div>
{:else}
  <a class="byline mono" href="https://adamsohn.com">adamsohn.com</a>
  <Layout>
    <header class="hero">
      <div class="kicker mono">Opening</div>
      <h1>CLAP, Contrastive Language-Audio Pretraining, visualized</h1>
      <p class="dek">CLIP, with another step for audio.</p>
      <HeroSphere />
      <p class="standfirst">
        CLAP is CLIP's trick applied to sound.
        <a href="https://github.com/LAION-AI/CLAP">LAION's Contrastive Language-Audio
        Pretraining</a> reuses the recipe that put images and text in one shared
        space, this time for audio and words. The goal is to tell whether a sound and
        a phrase mean the same thing, so each is turned into a fingerprint: a list of
        512 numbers. Two fingerprints that <em>point the same way</em> count as a
        match, even when the numbers differ. This page follows one engine recording
        through every operation that builds its fingerprint, then does the same for
        the words and measures the match. Every value is computed live in your browser
        from the real sound. The encoder's whole trick is to repeat two moves,
        <em>find patterns, then shrink</em>.
      </p>
      <p class="standfirst">
        One caveat up front: the encoder here is a <em>miniature</em>, trained from
        scratch on just these few clips so every number stays small enough to watch.
        The real LAION-CLAP runs far heavier machinery: a Swin vision transformer
        (HTSAT) reads the spectrogram for audio, RoBERTa handles the text, and it
        learns from hundreds of thousands of audio-caption pairs. This page keeps
        CLAP's exact recipe (spectrogram → encode → pool → project to 512 → normalise
        → compare by cosine) but swaps those encoders for plain convolutions and a
        tiny word-level transformer. Every step's shape is real and computed live;
        only the scale and the architecture are toy. The one piece of the genuine
        model on this page is the opening sphere: 2,600 real LAION-CLAP embeddings.
      </p>
      <div class="alsosee">
        <p class="also-label mono">Also see</p>
        <ul>
          <li>
            <a href="https://adamsohn.com/separate/"><strong>Isolating the engine
            audio</strong></a> — an interactive, scroll-driven tutorial. It takes a
            real recording where the engine is mixed in with music, a voice talking
            over it, or street noise, and pulls out just the diagnostic engine sound.
            The whole separation pipeline runs live (waveform → spectrogram, source
            separation, speech detection), and the clean, isolated clip drops out at
            the end.
          </li>
          <li>
            <a href="https://github.com/adam-s/car-diagnosis/"><strong>car-diagnosis</strong></a>
            — the repo behind the opening sphere: an end-to-end system that turns a
            shaky phone clip of a rattling car into a named fault. It scrapes fault
            recordings off the web, cleans them, embeds them with CLAP, trains
            classifiers on those embeddings, and serves an inference page where you
            drop in a recording and it tells you what's wrong.
          </li>
        </ul>
      </div>
    </header>

    <Section eyebrow="Pick a sound" title="The car-diagnosis clips">
      <Prose>
        <p>
          Choose an engine sound — or drop in your own clip. It's decoded and run
          through the encoder right here; every step below shows the real values
          for whatever you pick.
        </p>
      </Prose>
      <ClipPicker clips={bundle.clips} />
    </Section>

    <Section eyebrow="The pipeline" title="From a waveform to 512 numbers">
      <Prose>
        <p>
          It starts as a sound you can play and ends as a single vector. Each step
          is one operation in the chain; press play to watch it run, and read what
          it does and why it's there.
        </p>
      </Prose>
      {#if active}
        <div class="steps">
          <SpectrogramStep {bundle} clip={active} {transport} n={1} />
          <ConvStep {bundle} clip={active} stage="patch" n={2} />
          <HierarchyStep {bundle} clip={active} n={3} />
          <PoolStep {bundle} clip={active} n={4} />
          <ProjectionStep {bundle} clip={active} n={5} />
        </div>
      {:else}
        <div class="computing mono">computing the encoder…</div>
      {/if}
    </Section>

    <Section eyebrow="The other side" title="Words into the same space">
      <Prose>
        <p>
          A sound is now 512 numbers. For CLAP to compare it with language, words
          have to land in that <em>same</em> space. The text encoder is shorter than
          the audio one, and its first move is the one most worth seeing. Pick a
          label and follow it through.
        </p>
      </Prose>
      <TextPicker items={bundle.text} selected={textSel} onpick={(i) => (textSel = i)} />
      <div class="steps">
        <TokenLookupStep {bundle} item={bundle.text[textSel]} n={6} />
        <ContextPoolStep {bundle} item={bundle.text[textSel]} n={7} />
        <TextProjectStep {bundle} item={bundle.text[textSel]} n={8} />
      </div>
    </Section>

    <Section eyebrow="Both sides meet" title="Does the sound match the words?">
      <Prose>
        <p>
          Now the two encoders can finally talk. The sound and every label are points
          in one space, so the match is just the angle between them: one dot product,
          computed live.
        </p>
      </Prose>
      {#if active}
        <div class="steps">
          <MatchStep {bundle} clip={active} n={9} />
          <MatrixStep {bundle} clip={active} n={10} />
        </div>
      {/if}
    </Section>

    <Section eyebrow="Where this goes" title="A sound you can name, search, and flag">
      <Prose>
        <p>
          One engine recording became a point in 512-dimensional space; the words
          became another; a single dot product compares them. That is the whole of
          CLAP: two encoders trained together until a sound and its description point
          the same way.
        </p>
        <p>
          What follows needs no classifier trained for the task. Score a sound
          against a set of phrases and keep the nearest, and you have zero-shot
          classification: the basis for a tool that names a fault from a phone
          recording. Run the comparison the other way and it is retrieval, a typed
          query ranking thousands of clips or a clip returning its most likely words.
          And a sound that lands far from everything is one the model has never
          really heard, which is enough to flag the rattle nobody has named yet.
        </p>
        <p>
          None of this is specific to cars, or even to audio. Encode each modality
          into one shared space and compare by angle, and the same machinery drives
          CLIP for images and JEPA for video. That sphere you started with was 2,600
          of these embeddings, real car faults, each already a point among words it
          was never labelled with. The hard part was never the 512 numbers; it was
          training two very different encoders to agree on what <em>similar</em>
          means. After that, it is all geometry.
        </p>
      </Prose>
    </Section>
  </Layout>
  <FooterBar clips={bundle.clips} {transport} winSec={bundle.meta.win_sec ?? 1.5} />
{/if}

<style>
  .boot { min-height: 100vh; display: flex; align-items: center; justify-content: center; color: var(--ink-dim); }
  .boot.err { color: var(--pos); }
  .byline {
    position: fixed; top: var(--space-md); right: var(--space-lg); z-index: 30;
    font-size: 0.74rem; color: var(--ink-faint);
    border-bottom: 1px solid var(--line-strong); text-decoration: none;
  }
  .byline:hover { color: var(--accent); }
  .hero { display: flex; flex-direction: column; gap: var(--space-md); margin-top: var(--space-xl); }
  .kicker { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  h1 { margin: 0; font-size: clamp(1.8rem, 5vw, 2.75rem); }
  .dek { margin: 2px 0 0; color: var(--ink-dim); font-size: 1.24rem; font-style: italic; font-family: var(--font-serif); }
  .standfirst { color: var(--ink-dim); font-size: 1.12rem; line-height: 1.65; max-width: 60ch; margin: var(--space-xs) 0 0; }
  .standfirst em { color: var(--ink); font-style: italic; }
  .alsosee { margin-top: var(--space-lg); max-width: 62ch; }
  .also-label { margin: 0 0 8px; color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .alsosee ul { margin: 0; padding-left: 1.1em; display: flex; flex-direction: column; gap: 10px; }
  .alsosee li { color: var(--ink-dim); font-size: 1rem; line-height: 1.6; }
  .alsosee li strong { font-weight: 600; }
  .steps {
    display: flex; flex-direction: column;
    width: min(900px, 94vw);
    margin-left: 50%;
    transform: translateX(-50%);
  }
  .computing { color: var(--ink-faint); padding: var(--space-xl) 0; }

  /* ── Mobile-only layout (≤560px). Tablet/desktop untouched. Centralised here via
     :global so each step's scoped internals get the same treatment. ───────────── */
  @media (max-width: 560px) {
    /* the fixed byline floats over content on phones — drop it */
    .byline { display: none; }
    /* step header: let the right-hand "dims" caption fall to its own line instead
       of colliding with the eyebrow */
    :global(.steps .step > header) { flex-wrap: wrap; row-gap: 3px; }
    :global(.steps .step > header .dims) { margin-left: 0; flex-basis: 100%; }
    /* side-by-side figure rows stack into a clean centred column */
    :global(.steps .step .row),
    :global(.steps .step .flow),
    :global(.steps .step .finale),
    :global(.steps .step .pool),
    :global(.steps .step .lookup) {
      flex-direction: column;
      align-items: center;
    }
  }
</style>
