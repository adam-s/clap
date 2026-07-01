<script lang="ts">
  import type { Snippet } from 'svelte';
  type Props = { children: Snippet };
  let { children }: Props = $props();
</script>

<div class="layout">
  <main class="content">
    {@render children()}
  </main>
</div>

<style>
  .layout {
    min-height: 100vh;
    max-width: var(--maxw);
    margin: 0 auto;
  }
  .content {
    min-width: 0;
    padding: var(--space-3xl) var(--space-xl) calc(var(--space-3xl) + 64px);
    display: flex;
    flex-direction: column;
    gap: var(--space-2xl);
  }
  @media (max-width: 820px) {
    .content {
      padding: var(--space-xl) var(--space-md) calc(var(--space-3xl) + 56px);
      gap: var(--space-xl);
    }
  }
</style>
