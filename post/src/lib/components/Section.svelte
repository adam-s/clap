<script lang="ts">
  import type { Snippet } from 'svelte';
  type Props = { eyebrow?: string; title?: string; id?: string; children: Snippet };
  let { eyebrow, title, id, children }: Props = $props();
</script>

<section {id}>
  {#if eyebrow}<div class="eyebrow mono">{eyebrow}</div>{/if}
  {#if title}<h2>{title}</h2>{/if}
  <div class="body">{@render children()}</div>
</section>

<style>
  section {
    scroll-margin-top: 24px;
    display: flex;
    flex-direction: column;
    gap: var(--space-md);
  }
  .eyebrow {
    color: var(--ink-faint);
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  h2 {
    margin: 0;
    font-size: clamp(1.5rem, 4vw, 2rem);
  }
  .body {
    display: flex;
    flex-direction: column;
    gap: var(--space-lg);
  }
</style>
