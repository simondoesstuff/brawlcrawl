<script lang="ts">
	import type { Brawler } from '$lib/onnx/metadata';
	import { RARITY_COLORS } from '$lib/pick/constants';
	import { pickIsAlly, type TurnSchedule } from '$lib/pick/phase';

	interface DraftPick {
		isAlly: boolean;
		brawler: Brawler;
	}

	interface Props {
		allyBans: Brawler[];
		enemyBans: Brawler[];
		picks: DraftPick[];
		phase: number;
		allyFirst: boolean;
		finalMode: boolean;
		turnSchedule: TurnSchedule;
	}

	let { allyBans, enemyBans, picks, phase, allyFirst, finalMode, turnSchedule }: Props = $props();

	const pickSlots = $derived(
		Array.from({ length: 6 }, (_, i) => {
			const isAllySlot = pickIsAlly(i, allyFirst, turnSchedule);
			const pick = picks[i];
			const isCurrent = !finalMode && i === phase - 6;
			return { isAllySlot, pick, isCurrent };
		})
	);
</script>

<div class="board">
	<div class="row">
		<span class="label">Bans</span>
		<div class="team ally">
			<span class="team-label">Ally</span>
			{#each { length: 3 } as _, i (i)}
				{#if allyBans[i]}
					<span class="chip ban" style:--c={RARITY_COLORS[allyBans[i].rarity]}>{allyBans[i].name}</span>
				{:else}
					<span class="chip empty">□</span>
				{/if}
			{/each}
		</div>
		<div class="team enemy">
			<span class="team-label">Enemy</span>
			{#each { length: 3 } as _, i (i)}
				{#if enemyBans[i]}
					<span class="chip ban" style:--c={RARITY_COLORS[enemyBans[i].rarity]}>{enemyBans[i].name}</span>
				{:else}
					<span class="chip empty">□</span>
				{/if}
			{/each}
		</div>
	</div>

	{#if phase >= 6 || picks.length}
		<div class="row">
			<span class="label">Picks</span>
			<div class="picks">
				{#each pickSlots as slot, i (i)}
					{#if slot.pick}
						<span
							class="chip pick"
							class:ally-pick={slot.isAllySlot}
							class:enemy-pick={!slot.isAllySlot}
							style:--c={RARITY_COLORS[slot.pick.brawler.rarity]}
						>
							{slot.pick.brawler.name}
						</span>
					{:else if slot.isCurrent}
						<span class="chip current" class:ally-pick={slot.isAllySlot} class:enemy-pick={!slot.isAllySlot}>?</span>
					{:else}
						<span class="chip empty">□</span>
					{/if}
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	.board {
		display: flex;
		flex-direction: column;
		gap: 0.5em;
		font-size: 0.85rem;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.6em;
	}
	.label {
		opacity: 0.5;
		font-size: 0.75em;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		min-width: 3em;
	}
	.team,
	.picks {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.3em;
	}
	.team-label {
		font-size: 0.7em;
		opacity: 0.6;
		margin-right: 0.15em;
	}
	.chip {
		padding: 0.25em 0.55em;
		border-radius: 0.4em;
		font-weight: 600;
		white-space: nowrap;
	}
	.chip.ban {
		color: var(--c);
		background: color-mix(in oklab, var(--color-fg) 8%, transparent);
	}
	.chip.pick.ally-pick {
		color: var(--c);
		background: color-mix(in oklab, var(--color-primary) 20%, transparent);
	}
	.chip.pick.enemy-pick {
		color: var(--c);
		background: color-mix(in oklab, var(--color-failure) 18%, transparent);
	}
	.chip.current.ally-pick {
		background: color-mix(in oklab, var(--color-primary) 35%, transparent);
	}
	.chip.current.enemy-pick {
		background: color-mix(in oklab, var(--color-failure) 30%, transparent);
	}
	.chip.empty {
		opacity: 0.3;
	}
</style>
