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

# Build every data/ artifact that forward-depends on a newly trained
# terminal checkpoint + its crawl file: the draft model, winrates/pickrates,
# and both tier-list variants -- dependency-tracked through
# workflow/Snakefile (draft_train, winrates, pickrates, tier_lists,
# tier_lists_optimal), so already-current outputs are skipped. Run this
# after `just train`/`just draft-train` produce data/terminal_<crawl>/
# model.eqx for a new crawl. Does not touch web/static -- that's a separate,
# explicit step (`just web-data <crawl>`) once you're ready to deploy it.
crawl-artifacts crawl:
	uv run snakemake -s workflow/Snakefile --cores 1 \
		data/draft_{{crawl}}/draft_q_best.eqx \
		data/winrates_{{crawl}}.json \
		data/pickrates_{{crawl}}.json \
		data/tier_lists_{{crawl}}.json \
		data/tier_lists_optimal_{{crawl}}.json

# Continue training from an existing checkpoint on a new battles file,
# (e.g. data/model/model.eqx and data/crawl_myt2_20260916.json).
continue-training ckpt data *args:
	#!/usr/bin/env sh
	shift 2
	uv run geneus --init-from "{{ckpt}}" --battles-file "{{data}}" "$@"

# Regenerate web/static/{data,models} from data/, without snakemake -- a fresh
# checkout gives every file the same mtime, so snakemake's staleness check
# can't be trusted to skip the expensive Monte Carlo tier-list regen (see
# workflow/Snakefile's `tier_lists` rule). CI's build step and this recipe are
# the same commands for that reason.
#
# `crawl` names the one dataset backing everything here: its own
# crawl_<crawl>.json battles, terminal_<crawl>/model.eqx, and
# draft_<crawl>/draft_q_best.eqx -- winrates/pickrates z-scores are always
# computed against the same crawl that trained the checkpoints, never a
# separately-pinned one.
#
# winrates/pickrates are cheap, deterministic aggregations over the crawl's
# own committed battles file -- regenerated fresh every time rather than
# committed. events.json and tier_lists*.json are NOT: events.json comes from
# a rate-limited, best-effort live crawl (needs BSTOK, isn't reproducible on
# demand), and tier_lists*.json are expensive Monte Carlo output -- both stay
# committed artifacts, just copied here.
web-data crawl="myt2_20260916":
	uv run compute-stats winrates --input data/crawl_{{crawl}}.json --output data/winrates_{{crawl}}.json
	uv run compute-stats pickrates --input data/crawl_{{crawl}}.json --output data/pickrates_{{crawl}}.json
	uv run export-onnx \
		--terminal-ckpt data/terminal_{{crawl}}/model.eqx \
		--draft-ckpt data/draft_{{crawl}}/draft_q_best.eqx \
		--stats-leg {{crawl}}
	cp data/tier_lists_{{crawl}}.json web/static/data/tier_lists.json
	cp data/tier_lists_optimal_{{crawl}}.json web/static/data/tier_lists_optimal.json

web-test: web-data
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
