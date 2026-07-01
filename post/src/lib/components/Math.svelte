<script lang="ts">
  /** KaTeX wrapper (grpo style). `\htmlClass{part-X}{…}` tags formula parts;
   * `active` dims all but `part-<active>` so the symbol lines up with the
   * current term of the animation. */
  // KaTeX (~268 kB) is code-split out of the initial bundle and loaded on mount;
  // every formula sits below the fold, so nothing renders late that the reader sees.
  import { onMount } from 'svelte';
  import 'katex/dist/katex.min.css';

  type Props = { tex: string; displayMode?: boolean; active?: string | null };
  let { tex, displayMode = false, active = null }: Props = $props();

  let katex = $state<typeof import('katex').default | null>(null);
  onMount(async () => { katex = (await import('katex')).default; });

  const html = $derived(
    katex
      ? katex.renderToString(tex, {
          displayMode,
          throwOnError: false,
          output: 'html',
          strict: 'ignore',
          trust: (ctx) => ctx.command === '\\htmlClass',
        })
      : '',
  );

  let host: HTMLElement | undefined = $state();
  $effect(() => {
    if (!host) return;
    void html;
    host.querySelectorAll<HTMLElement>('[class*="part-"]').forEach((el) => {
      el.style.opacity = active == null || el.classList.contains(`part-${active}`) ? '1' : '0.3';
    });
  });
</script>

<span class="math" class:block={displayMode} bind:this={host}>{@html html}</span>

<style>
  .math :global(.katex) { color: inherit; }
  .math.block { display: flex; justify-content: center; }
  .math :global([class*='part-']) { transition: opacity 160ms ease; }
</style>
