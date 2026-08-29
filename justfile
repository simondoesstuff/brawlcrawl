_default:
	@just --list

crawl *args="--help":
	uv run crawl {{args}}

pick *args="":
	uv run pick {{args}}

typecheck path='src':
	uv run basedpyright {{path}}

alias t := test
test *args:
	just typecheck && echo ''
	uv run pytest {{args}}

alias i := install
install:
	uv sync
