_default:
	@just --list

run *args="--help":
	uv run brawl {{args}}

typecheck path='src':
	uv run basedpyright {{path}}

alias t := test
test *args:
	just typecheck && echo ''
	uv run pytest {{args}}

alias i := install
install:
	uv sync
