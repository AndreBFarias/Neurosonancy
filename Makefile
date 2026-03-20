.PHONY: sync-voices validate-voices sync package-release

LUNA_PATH ?= ../Luna
VERSION ?= v1.0.0

sync-voices:
	python scripts/sync_voices.py --luna-path $(LUNA_PATH)

validate-voices:
	python scripts/validate_voices.py --luna-path $(LUNA_PATH)

sync: sync-voices validate-voices

package-release:
	python scripts/package_release.py --version $(VERSION)
