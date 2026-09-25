set positional-arguments := true

_default:
	@just --list

crawl *args="--help":
	#!/usr/bin/env sh
	caffeinate -i uv run crawl "$@"

pick *args="":
	#!/usr/bin/env sh
	uv run pick "$@"

train *args="":
	#!/usr/bin/env sh
	uv run geneus "$@"

draft-train terminal out *args:
	#!/usr/bin/env sh
	shift 2
	uv run draft-train train --terminal-ckpt {{terminal}} --out {{out}} "$@"

# Continue training from an existing checkpoint on a new battles file,
# (e.g. data/model/model.eqx and data/crawl_myt2_20260916.json).
continue-training ckpt data *args:
	#!/usr/bin/env sh
	shift 2
	uv run geneus --init-from "{{ckpt}}" --battles-file "{{data}}" "$@"

export-onnx:
	snakemake --cores 1 export_onnx web/static/data/tier_lists.json

# Regenerate web/static/{data,models} from data/, without snakemake -- a fresh
# checkout gives every file the same mtime, so snakemake's staleness check
# can't be trusted to skip the expensive Monte Carlo tier-list regen (see
# workflow/Snakefile's `tier_lists` rule). CI's build step and this recipe are
# the same commands for that reason.
#
# winrates/pickrates are cheap, deterministic aggregations over the committed
# crawl_leg1_20260827.json -- regenerated fresh every time rather than
# committed. events.json and tier_lists*.json are NOT: events.json comes from
# a rate-limited, best-effort live crawl (needs BSTOK, isn't reproducible on
# demand), and tier_lists*.json are expensive Monte Carlo output -- both stay
# committed artifacts, just copied here.
web-data:
	uv run compute-stats winrates --input data/crawl_leg1_20260827.json --output data/winrates_leg1_20260827.json
	uv run compute-stats pickrates --input data/crawl_leg1_20260827.json --output data/pickrates_leg1_20260827.json
	uv run export-onnx \
		--terminal-ckpt data/terminal_myt2_20260916/model.eqx \
		--draft-ckpt data/draft_myt2_20260916/draft_q_best.eqx
	cp data/tier_lists.json web/static/data/tier_lists.json
	cp data/tier_lists_optimal.json web/static/data/tier_lists_optimal.json

web-test: export-onnx
	cd web && bun test

typecheck path='src':
	uv run basedpyright {{path}}

alias t := test
test *args:
	just typecheck && echo ''
	uv run pytest {{args}}

alias i := install
install:
	uv sync
