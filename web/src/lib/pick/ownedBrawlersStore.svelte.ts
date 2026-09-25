// Persisted, app-global raw text for the owned-brawlers setting (see
// ownedBrawlers.ts for resolution against the roster). A single source of
// truth shared by the settings page and the in-draft filter panel.

const STORAGE_KEY = 'brawl:ownedBrawlersText';

function loadStoredText(): string {
	if (typeof localStorage === 'undefined') return '';
	try {
		return localStorage.getItem(STORAGE_KEY) ?? '';
	} catch {
		return '';
	}
}

class OwnedBrawlersStore {
	text = $state(loadStoredText());

	setText(next: string): void {
		this.text = next;
		try {
			localStorage.setItem(STORAGE_KEY, next);
		} catch {
			/* best-effort persistence only */
		}
	}
}

export const ownedBrawlers = new OwnedBrawlersStore();
