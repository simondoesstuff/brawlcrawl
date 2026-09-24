// Types for web/static/data/tier_lists.json, written by `uv run tier-list`
// (src/pick/tier_list.py::_map_result_to_json) and copied into place by the
// `tier_lists_web` Snakemake rule. `tier_list` is pre-sorted by win_rate desc.

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
	maps: MapTierList[];
}
