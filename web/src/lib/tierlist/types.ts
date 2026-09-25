// Types for web/static/data/tier_lists.json and tier_lists_optimal.json,
// written by `uv run tier-list --policy {random,optimal}`
// (src/pick/tier_list.py::_map_result_to_json) into data/tier_lists_<crawl>.json
// / data/tier_lists_optimal_<crawl>.json (Snakemake `tier_lists` /
// `tier_lists_optimal` rules) and copied into place by `just web-data`.
// `tier_list` is pre-sorted by win_rate desc. `policy` distinguishes the two
// files: "random" (uniform continuation) vs "optimal" (DraftQNetwork continuation).

export interface TierEntry {
	brawler_id: number;
	name: string;
	win_rate: number;
	samples: number;
}

export interface MapTierList {
	event_id: number;
	map_name: string;
	mode: string;
	trials: number;
	converged: boolean;
	tier_list: TierEntry[];
}

export interface TierListsData {
	policy: 'random' | 'optimal';
	maps: MapTierList[];
}
