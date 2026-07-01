<script lang="ts">
  import type { Snippet } from 'svelte';
  type Props = { children: Snippet };
  let { children }: Props = $props();
</script>

<div class="prose">{@render children()}</div>

<style>
  .prose {
    color: var(--ink-dim);
    font-size: 1.06rem;
    line-height: 1.6;
    max-width: 64ch;
  }
  .prose :global(p) { margin: 0 0 var(--space-md); }
  .prose :global(p:last-child) { margin-bottom: 0; }
  .prose :global(strong) { color: var(--ink); font-weight: 620; }
  .prose :global(em) { color: var(--ink); font-style: normal; border-bottom: 1px dotted var(--ink-faint); }
  .prose :global(code) {
    font-family: ui-monospace, Menlo, monospace;
    font-size: 0.9em;
    background: var(--panel-2);
    padding: 1px 5px;
    border-radius: 4px;
    color: var(--ink);
  }
</style>
