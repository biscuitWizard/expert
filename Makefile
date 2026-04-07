.PHONY: setup fmt check

setup:
	git config core.hooksPath .githooks

fmt:
	cargo fmt --all

check:
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings
