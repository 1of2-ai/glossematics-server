# GlossematicsServer — OpenAI-compatible /v1/embeddings localhost daemon.
#
# Self-contained: depends only on the Glossematics inference library via a path dependency
# into the sibling GlossematicsSDK checkout (see Package.swift). Bundles are passed by path —
# `make dummy-bundle` generates a no-op test bundle locally, so everything works without any
# conversion pipeline on the machine. Point BUNDLE= at a real bundle for real embeddings.

# ----- configuration ---------------------------------------------------------

BUNDLE      ?= TestBundles/JinaV5OmniSmall.w8a16.dummy.bundle
SOURCE_BUNDLE ?=                    # real bundle the dummy generator mirrors (see dummy-bundle)
PYTHON      ?= python3              # must have coremltools + torch + numpy for dummy-bundle
PORT        ?= 11435
DIMENSIONS  ?= 1024
MODEL_NAME  ?=
LABEL       ?= com.meridian.glossematics.embeddings
LOG_FILE    ?= $(HOME)/Library/Logs/Glossematics/embeddings-server.log

# client probes against a running server
MODEL       ?= jinaai/jina-embeddings-v5-omni-small
TEXT        ?= What is the capital of France?
DIMS        ?= 512
TASK        ?=
BASE_URL    ?= http://127.0.0.1:$(PORT)

# export
EXPORT_DIR    ?= dist
EXPORT_PREFIX ?= glossematics-server
VERSION       := $(shell git describe --always --dirty 2>/dev/null || echo dev)
ARCHIVE       := $(EXPORT_DIR)/$(EXPORT_PREFIX)-$(VERSION).zip

SERVER_BIN := $(shell swift build --show-bin-path 2>/dev/null)/gloss-server

.DEFAULT_GOAL := help
.PHONY: help build release test dummy-bundle run install uninstall restart status \
        logs health models embed export clean

# ----- targets ---------------------------------------------------------------

help: ## list targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## debug-build the server
	swift build

release: ## release-build the server
	swift build -c release

test: ## run server tests (docs packaging + templating)
	swift test

dummy-bundle: ## generate the no-op test bundle: make dummy-bundle SOURCE_BUNDLE=/path/to/real.bundle
	@$(PYTHON) -c "import coremltools, torch, numpy" 2>/dev/null || { \
		echo "error: $(PYTHON) lacks coremltools/torch/numpy."; \
		echo "  pip install coremltools torch numpy   # or point PYTHON= at a venv that has them"; \
		exit 1; }
	@test -n "$(SOURCE_BUNDLE)" || { \
		echo "error: set SOURCE_BUNDLE=<real native-v1 bundle dir>"; \
		echo "  the dummy mirrors its tokenizer, vision assets, and compiled I/O schemas"; \
		exit 1; }
	@test -d "$(SOURCE_BUNDLE)" || { echo "error: source bundle not found: $(SOURCE_BUNDLE)"; exit 1; }
	@mkdir -p TestBundles
	$(PYTHON) Scripts/make_dummy_bundle.py --source "$(SOURCE_BUNDLE)" --output "$(BUNDLE)" --force

run: build ## run gloss-server in the foreground (BUNDLE defaults to the generated dummy)
	@test -d "$(BUNDLE)" || { \
		echo "error: bundle not found: $(BUNDLE)"; \
		echo "  generate the no-op test bundle:  make dummy-bundle SOURCE_BUNDLE=<real bundle>"; \
		echo "  or point BUNDLE= at a real bundle"; \
		exit 1; }
	@args=""; [ -n "$(MODEL_NAME)" ] && args="--model-name $(MODEL_NAME)"; \
	$(SERVER_BIN) --bundle "$(BUNDLE)" --port "$(PORT)" --dimensions "$(DIMENSIONS)" $$args

install: ## build release, install, and start the launchd agent (BUNDLE should be a REAL bundle)
	@if [[ "$(BUNDLE)" == *dummy* && "$(ALLOW_DUMMY)" != "1" ]]; then \
		echo "error: refusing to install the launch agent with a dummy bundle (constant embeddings)."; \
		echo "  set BUNDLE=<real bundle>, or ALLOW_DUMMY=1 to override."; \
		exit 1; fi
	@test -d "$(BUNDLE)" || { echo "error: bundle not found: $(BUNDLE)"; exit 1; }
	@args=""; [ -n "$(MODEL_NAME)" ] && args="--model-name $(MODEL_NAME)"; \
	Scripts/gloss-server-launchagent.sh install \
		--bundle "$(BUNDLE)" --port "$(PORT)" --dimensions "$(DIMENSIONS)" --label "$(LABEL)" $$args

uninstall: ## stop and remove the launchd agent
	Scripts/gloss-server-launchagent.sh uninstall --label "$(LABEL)"

restart: ## restart the launchd agent (reload plist changes)
	Scripts/gloss-server-launchagent.sh restart --label "$(LABEL)"

status: ## agent state + endpoint health
	Scripts/gloss-server-launchagent.sh status --label "$(LABEL)"

logs: ## tail the agent log
	@test -f "$(LOG_FILE)" || { echo "no log at $(LOG_FILE)"; exit 1; }
	tail -f "$(LOG_FILE)"

health: ## GET /health from the running server
	@curl -fsS "$(BASE_URL)/health" | python3 -m json.tool

models: ## GET /v1/models from the running server
	@curl -fsS "$(BASE_URL)/v1/models" | python3 -m json.tool

embed: ## POST one embedding: make embed TEXT="..." DIMS=512 [TASK=retrieval.query]
	@python3 -c 'import json,sys; m,t,d,task=sys.argv[1:5]; \
		p={"model":m,"input":t,"dimensions":int(d)}; p.update({"task":task} if task else {}); \
		print(json.dumps(p))' "$(MODEL)" "$(TEXT)" "$(DIMS)" "$(TASK)" \
	| curl -fsS -m 300 "$(BASE_URL)/v1/embeddings" -H 'Content-Type: application/json' -d @- \
	| python3 -c 'import json,sys,math; r=json.load(sys.stdin); \
		(print(json.dumps(r["error"],indent=2)), sys.exit(1)) if "error" in r else None; \
		e=r["data"][0]["embedding"]; \
		print("dims", len(e), "norm %.4f" % math.sqrt(sum(x*x for x in e)), "usage", r["usage"])'

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

clean: ## remove build dirs, test bundles, and dist/
	@rm -rf .build "$(EXPORT_DIR)" TestBundles
	@echo "cleaned build dirs, TestBundles/, and $(EXPORT_DIR)/"
