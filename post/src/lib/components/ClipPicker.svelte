<script lang="ts">
  /**
   * ClipPicker — the top-of-page sound switcher. It reflects the SHARED activeClip
   * (single source of truth), so it stays in sync no matter where the sound was
   * switched from (here or the footer bar). No local "which is selected" state.
   */
  import { onDestroy } from 'svelte';
  import type { Clip } from '../data/clapData';
  import { activeClip, selectDemo, loadFile } from '../data/clapLive.svelte';

  type Props = { clips: Clip[] };
  let { clips }: Props = $props();

  let active = $state<Clip | null>(null);
  const u = activeClip.subscribe((c) => (active = c));
  onDestroy(u);

  const uploadName = $derived(active?.slug === 'your clip' ? active.caption : null);
  let errMsg = $state<string | null>(null);

  const MAX_MB = 25;
  const AUDIO_EXT = /\.(mp3|wav|m4a|aac|ogg|oga|opus|flac|webm)$/i;

  function onFile(e: Event) {
    const input = e.target as HTMLInputElement;
    const f = input.files?.[0];
    input.value = ''; // allow re-picking the same file after an error
    if (!f) return;
    errMsg = null;
    if (!(f.type.startsWith('audio/') || AUDIO_EXT.test(f.name))) {
      errMsg = 'Please choose an audio file (mp3, wav, m4a, ogg, flac…).';
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      errMsg = `That file is ${(f.size / 1048576).toFixed(0)} MB — max ${MAX_MB} MB.`;
      return;
    }
    loadFile(f).catch(() => {
      errMsg = "Couldn't decode that file — try a standard mp3 or wav.";
      selectDemo(0);
    });
  }
</script>

<div class="picker">
  {#each clips as c, i (c.slug)}
    <button class:active={c.slug === active?.slug} onclick={() => selectDemo(i)}>{c.caption}</button>
  {/each}
  <label class="upload" class:active={active?.slug === 'your clip'}>
    {uploadName ? `▶ ${uploadName}` : '＋ your own clip'}
    <input type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac" onchange={onFile} />
  </label>
</div>
{#if errMsg}<p class="err mono">{errMsg}</p>{/if}

<style>
  .picker { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  button, .upload {
    background: var(--panel-2);
    border: 1px solid var(--line);
    color: var(--ink-dim);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.82rem;
    cursor: pointer;
  }
  button.active, .upload.active {
    border-color: var(--accent);
    color: var(--ink);
    background: color-mix(in srgb, var(--accent) 16%, var(--panel-2));
  }
  .upload { border-style: dashed; }
  .upload input { display: none; }
  .err { margin: 8px 0 0; color: var(--pos); font-size: 0.78rem; }
</style>
