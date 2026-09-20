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

# Continue training from an existing checkpoint on a new battles file,
# resetting the optimizer/scheduler. ckpt and data are both relative to data/
# (e.g. model/model.eqx and crawl_myt2_20260916.json).
continue-training ckpt data *args="":
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
