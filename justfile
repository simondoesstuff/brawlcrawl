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
	snakemake --cores 1 export_onnx

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
