# GlossematicsServer — OpenAI-compatible /v1/embeddings localhost daemon for BidirLM-Omni and
# jina-embeddings-v5-omni-small bundles (the family is detected from the bundle manifest).
#
# The fixture is a real compiled Core ML bundle with the BidirLM contract and no-op functions.
# Production bundles come from GlossematicsCoreML (bidirlm/ releases, artifacts/JinaV5OmniSmall.*).

# ----- configuration ---------------------------------------------------------

FIXTURE     ?= Fixtures/BidirLMOmni.dummy.bundle
JINA_FIXTURE ?= Fixtures/JinaV5OmniSmall.w8a16.dummy.bundle
JINA_SOURCE ?= ../GlossematicsCoreML/artifacts/JinaV5OmniSmall.w8a16.bundle
BUNDLE      ?= $(FIXTURE)
COMPUTE     ?= ane
PYTHON      ?= python3
CONVERTER_PYTHON ?= ../GlossematicsCoreML/bidirlm/.venv/bin/python
TOKENIZER   ?=
ALLOW_DUMMY ?= 0
PORT        ?= 11435
MODEL_NAME  ?=
LABEL       ?= com.meridian.glossematics.embeddings
LOG_FILE    ?= $(HOME)/Library/Logs/Glossematics/embeddings-server.log
MAX_BATCH   ?= 2048
MAX_BODY_MB ?= 64
MAX_TOTAL_BODY_MB ?= 256
MAX_QUEUE_REQUESTS ?= 128
MAX_QUEUE_ITEMS ?= 8192
MAX_REQUEST_TOKENS ?= 131072
ANE_PROGRAM_BUDGET ?= 64
BATCH_WINDOW_MS ?= 2.0
KEEP_WARM_SECONDS ?= 60
MAX_CONNECTIONS ?= 256
IO_TIMEOUT_SECONDS ?= 30
SHUTDOWN_GRACE_SECONDS ?= 15
ACCESS_LOG ?= errors
MAX_TOKENS ?= 32768
DIMENSIONS ?=
JINA_REFERENCE ?= ../GlossematicsCoreML/reference
ORACLE ?=

# bidirlm | jina | unknown, from the bundle manifest
FAMILY := $(shell python3 -c 'import json,sys; m=json.load(open(sys.argv[1]+"/manifest.json")); print("bidirlm" if str(m.get("format","")).startswith("bidirlm-omni") else "jina" if m.get("formatVersion")==2 else "unknown")' "$(BUNDLE)" 2>/dev/null || echo unknown)

# client probes against a running server
MODEL       ?= $(if $(filter jina,$(FAMILY)),jinaai/jina-embeddings-v5-omni-small,BidirLM/BidirLM-Omni-2.5B-Embedding)
TEXT        ?= What is the capital of France?
BASE_URL    ?= http://127.0.0.1:$(PORT)

# export
EXPORT_DIR    ?= dist
EXPORT_PREFIX ?= glossematics-server
VERSION       := $(shell git describe --always --dirty 2>/dev/null || echo dev)
ARCHIVE       := $(EXPORT_DIR)/$(EXPORT_PREFIX)-$(VERSION).zip

SERVER_BIN := $(shell swift build --show-bin-path 2>/dev/null)/gloss-server
RELEASE_BIN := $(shell swift build -c release --show-bin-path 2>/dev/null)/gloss-server
FIXTURE_TABLE := $(FIXTURE)/token_embeddings.f16

# Placement flags are BidirLM-only; Jina bundles take an optional default Matryoshka size.
FAMILY_ARGS = $(if $(filter jina,$(FAMILY)),,--compute "$(COMPUTE)" --ane-program-budget "$(ANE_PROGRAM_BUDGET)") \
	$(if $(DIMENSIONS),--dimensions "$(DIMENSIONS)",)

SERVER_ARGS = --bundle "$(BUNDLE)" --port "$(PORT)" $(FAMILY_ARGS) \
	--max-batch "$(MAX_BATCH)" --max-body-mb "$(MAX_BODY_MB)" \
	--max-total-body-mb "$(MAX_TOTAL_BODY_MB)" \
	--max-queue-requests "$(MAX_QUEUE_REQUESTS)" --max-queue-items "$(MAX_QUEUE_ITEMS)" \
	--max-request-tokens "$(MAX_REQUEST_TOKENS)" \
	--batch-window-ms "$(BATCH_WINDOW_MS)" --keep-warm-seconds "$(KEEP_WARM_SECONDS)" \
	--max-connections "$(MAX_CONNECTIONS)" --idle-timeout-seconds "$(IO_TIMEOUT_SECONDS)" \
	--shutdown-grace-seconds "$(SHUTDOWN_GRACE_SECONDS)" --access-log "$(ACCESS_LOG)"

.DEFAULT_GOAL := help
.PHONY: help build release fixture test test-release verify-model verify-jina dummy-bundle jina-fixture package run install uninstall \
        restart status logs health live metrics models embed bench check-config export clean

# ----- targets ---------------------------------------------------------------

help: ## list targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

build: ## debug-build the server
	swift build

release: ## release-build the server
	swift build -c release

$(FIXTURE_TABLE):
	@echo "writing the fixture's all-zero token table (not checked in)"
	@dd if=/dev/zero of="$@" bs=4096 count=151936 status=none

fixture: $(FIXTURE_TABLE) ## write the fixture's zero token table (622 MB, gitignored)

test: fixture ## unit tests + debug/release golden-fixture HTTP smoke (BidirLM and Jina fixtures)
	swift test
	$(PYTHON) Scripts/test_server_http.py --server-bin "$(SERVER_BIN)" --bundle "$(FIXTURE)"
	$(PYTHON) Scripts/test_jina_http.py --server-bin "$(SERVER_BIN)" --bundle "$(JINA_FIXTURE)"
	$(MAKE) test-release

test-release: release fixture ## golden-fixture HTTP smoke against the optimized binary
	$(PYTHON) Scripts/test_server_http.py --server-bin "$(RELEASE_BIN)" --bundle "$(FIXTURE)"
	$(PYTHON) Scripts/test_jina_http.py --server-bin "$(RELEASE_BIN)" --bundle "$(JINA_FIXTURE)"

jina-fixture: ## regenerate the Jina no-op fixture from JINA_SOURCE (needs coremltools, torch: CONVERTER_PYTHON)
	$(CONVERTER_PYTHON) Scripts/make_jina_dummy_bundle.py --source "$(JINA_SOURCE)" --output "$(JINA_FIXTURE)" --force

package: ## signed disk image in dist/ (CODESIGN_IDENTITY=, NOTARIZE=1 with NOTARY_KEY_PATH/ID/ISSUER_ID)
	Scripts/package_release.sh --identity "$${CODESIGN_IDENTITY:--}" $(if $(filter 1,$(NOTARIZE)),--notarize,)

verify-model: release ## full-model gates on a sealed bundle: FP32 parity + HTTP (set BUNDLE=, COMPUTE=)
	$(RELEASE_BIN) --bundle "$(BUNDLE)" --compute "$(COMPUTE)" --check-config
	GLOSS_BIDIRLM_BUNDLE="$(abspath $(BUNDLE))" GLOSS_BIDIRLM_COMPUTE="$(COMPUTE)" \
		GLOSS_BIDIRLM_MAX_TOKENS="$(MAX_TOKENS)" swift test --filter "bidirlm|mediaPreprocessing"
	$(PYTHON) Scripts/test_server_http.py --server-bin "$(RELEASE_BIN)" --bundle "$(BUNDLE)" \
		--compute "$(COMPUTE)" --full-model

dummy-bundle: ## regenerate the no-op fixture (needs TOKENIZER=<bundle>/tokenizer and coremltools)
	@test -n "$(TOKENIZER)" || { echo "error: set TOKENIZER to a BidirLM bundle's tokenizer directory"; exit 1; }
	$(CONVERTER_PYTHON) Scripts/make_dummy_bundle.py --tokenizer "$(TOKENIZER)" --output "$(FIXTURE)" --force

verify-jina: release ## Jina gates on BUNDLE: video goldens + HTTP parity (ORACLE=<old daemon URL> optional)
	$(RELEASE_BIN) --bundle "$(BUNDLE)" --check-config
	GLOSS_JINA_BUNDLE="$(abspath $(BUNDLE))" GLOSS_JINA_REFERENCE="$(abspath $(JINA_REFERENCE))" \
		GLOSS_PRODUCTION_BUNDLE="$(abspath $(BUNDLE))" \
		swift test --filter "JinaFullModel|productionModelRetrievalEndToEnd"
	cd Scripts && $(CONVERTER_PYTHON) smoke_jina.py --server-bin "$(RELEASE_BIN)" --bundle "$(abspath $(BUNDLE))" \
		--output "$(abspath dist/jina-smoke)" --video-reference ../reference/jina/video_reference.json \
		$(if $(ORACLE),--oracle "$(ORACLE)",)

run: build ## run gloss-server in the foreground (ALLOW_DUMMY=1 for the fixture; COMPUTE=ane|gpu|cpu)
	@test -d "$(BUNDLE)" || { echo "error: bundle not found: $(BUNDLE)"; exit 1; }
	@if [ "$(BUNDLE)" = "$(FIXTURE)" ]; then $(MAKE) --no-print-directory fixture; fi
	@set -- $(SERVER_ARGS); \
	[ -z "$(MODEL_NAME)" ] || set -- "$$@" --model-name "$(MODEL_NAME)"; \
	[ "$(ALLOW_DUMMY)" != "1" ] || set -- "$$@" --allow-dummy; \
	$(SERVER_BIN) "$$@"

check-config: build ## validate the configuration and bundle, then exit
	@test -d "$(BUNDLE)" || { echo "error: bundle not found: $(BUNDLE)"; exit 1; }
	@set -- $(SERVER_ARGS) --check-config; \
	[ -z "$(MODEL_NAME)" ] || set -- "$$@" --model-name "$(MODEL_NAME)"; \
	[ "$(ALLOW_DUMMY)" != "1" ] || set -- "$$@" --allow-dummy; \
	$(SERVER_BIN) "$$@"

install: ## build release, install, and start the launchd agent (BUNDLE must be a sealed bundle)
	@test -d "$(BUNDLE)" || { echo "error: bundle not found: $(BUNDLE)"; exit 1; }
	@set -- install $(SERVER_ARGS) --label "$(LABEL)"; \
	[ -z "$(MODEL_NAME)" ] || set -- "$$@" --model-name "$(MODEL_NAME)"; \
	ALLOW_DUMMY="$(ALLOW_DUMMY)" Scripts/gloss-server-launchagent.sh "$$@"

uninstall: ## stop and remove the launchd agent
	Scripts/gloss-server-launchagent.sh uninstall --label "$(LABEL)"

restart: ## restart the launchd agent (reload plist changes)
	Scripts/gloss-server-launchagent.sh restart --label "$(LABEL)"

status: ## agent state + endpoint health
	Scripts/gloss-server-launchagent.sh status --label "$(LABEL)"

logs: ## tail the agent log
	@test -f "$(LOG_FILE)" || { echo "no log at $(LOG_FILE)"; exit 1; }
	tail -f "$(LOG_FILE)"

health: ## GET /ready from the running server
	@curl -sS "$(BASE_URL)/ready" | python3 -m json.tool

live: ## GET /live from the running server
	@curl -fsS "$(BASE_URL)/live" | python3 -m json.tool

metrics: ## GET Prometheus metrics from the running server
	@curl -fsS "$(BASE_URL)/metrics"

models: ## GET /v1/models from the running server
	@curl -fsS "$(BASE_URL)/v1/models" | python3 -m json.tool

embed: ## POST one text embedding: make embed TEXT="..."
	@python3 -c 'import json,sys; m,t=sys.argv[1:3]; print(json.dumps({"model":m,"input":t}))' "$(MODEL)" "$(TEXT)" \
	| curl -fsS -m 900 "$(BASE_URL)/v1/embeddings" -H 'Content-Type: application/json' -d @- \
	| python3 -c 'import json,sys,math; r=json.load(sys.stdin); \
		(print(json.dumps(r["error"],indent=2)), sys.exit(1)) if "error" in r else None; \
		e=r["data"][0]["embedding"]; \
		print("dims", len(e), "norm %.4f" % math.sqrt(sum(x*x for x in e)), "usage", r["usage"])'

bench: ## benchmark concurrent text throughput and packing
	@python3 Scripts/benchmark_batching.py --base-url "$(BASE_URL)" --model "$(MODEL)" \
		--concurrency "$${CONCURRENCY:-16}" --requests "$${REQUESTS:-200}" --items "$${ITEMS:-1}" \
		--tokens "$${TOKENS:-24}" --warmup-requests "$${WARMUP_REQUESTS:-20}"

export: ## dist/$(EXPORT_PREFIX)-<version>.zip from git-tracked files (EXCLUDE="glob ..." to drop paths)
	@mkdir -p "$(EXPORT_DIR)"
	@rm -f "$(ARCHIVE)"
	@git ls-files --cached --others --exclude-standard -z | xargs -0 zip -q "$(ARCHIVE)"
ifneq ($(strip $(EXCLUDE)),)
	@zip -q -d "$(ARCHIVE)" $(foreach pat,$(EXCLUDE),'$(pat)') || [ $$? -eq 12 ]
endif
	@echo "exported -> $(ARCHIVE)"
	@unzip -l "$(ARCHIVE)" | tail -1 | awk '{printf "  %s files, %.1f MB uncompressed\n", $$2, $$1/1048576}'
	@ls -lh "$(ARCHIVE)" | awk '{print "  archive size: " $$5}'

clean: ## remove build dirs, the fixture token table, and dist/
	@rm -rf .build "$(EXPORT_DIR)" TestBundles "$(FIXTURE_TABLE)"
	@echo "cleaned build dirs, the fixture token table, and $(EXPORT_DIR)/"
