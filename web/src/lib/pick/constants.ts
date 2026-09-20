// Port of src/pick/constants.py — keep values in sync with that file.

/** Pickrate z-score thresholds (relative to per-event mean). */
export const PICKRATE_HIGH_Z = 1.0; // "often picked" — top ~10% of pickrates
export const PICKRATE_LOW_Z = -0.5; // "rarely picked" — below-average pickrate

/** After z-scoring over available brawlers: z > 0 = above-average for this draft state. */
export const Q_THRESHOLD = 0.0;

export const ANN_BRAIN = 'brain'; // Q > 0 and low pickrate (hidden gem)
export const ANN_SECRET = 'secret'; // overview: good map score, low pickrate (secret pick)
export const ANN_X = 'overrated'; // Q <= 0 or bad map score, high pickrate (overrated)

/** Port of src/pick/display.py::RARITY_COLORS (CSS-safe hex, tuned for both themes). */
export const RARITY_COLORS: Record<string, string> = {
	'Starting Brawler': '#c9ccd6',
	Rare: '#3ecf5f',
	'Super Rare': '#00afff',
	Epic: '#b060ff',
	Mythic: '#ff4d4d',
	Legendary: '#e8c400',
	'Ultra Legendary': '#fff066'
};

export const RARITY_ORDER = [
	'Starting Brawler',
	'Rare',
	'Super Rare',
	'Epic',
	'Mythic',
	'Legendary',
	'Ultra Legendary'
];

export const ANNOTATION_LEGEND: { key: string; symbol: string; label: string }[] = [
	{ key: ANN_SECRET, symbol: '🤫', label: 'secret pick' },
	{ key: ANN_BRAIN, symbol: '🧠', label: 'genius AI pick' },
	{ key: ANN_X, symbol: '✗', label: 'popular, but bad' }
];
