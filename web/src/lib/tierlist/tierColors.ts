import type { Tier } from './tiers';

/** Display-only badge colors per tier, distinct from RARITY_COLORS to avoid confusion. */
export const TIER_COLORS: Record<Tier, string> = {
	S: '#ff4d4d',
	A: '#ff9f40',
	B: '#ffd93d',
	C: '#6bcb77',
	D: '#4d96ff'
};
