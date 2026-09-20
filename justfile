set positional-arguments := true

_default:
	@just --list

crawl *args="--help":
	#!/usr/bin/env sh
	caffeinate -i uv run crawl "$@"

pick *args="":
	#!/usr/bin/env sh
	uv run pick "$@"

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
