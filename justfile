# Default: list available recipes
default:
    @just --list

# Make the project root importable for `python -m loader`
export PYTHONPATH := "."

# Run the data loader (all options are optional)
# Usage: just load
#        just load env=prod batch-size=500 reset=true max-records=1000 no-embed=true
load env="local" batch-size="1000" reset="false" max-records="0" no-embed="false":
    uv run python -m loader load \
        --env {{env}} \
        --batch-size {{batch-size}} \
        {{if reset == "true" { "--reset" } else { "" }}} \
        --max-records {{max-records}} \
        {{if no-embed == "true" { "--no-embed" } else { "" }}}

# Dry-run: load without embedding generation
load-dry env="local" batch-size="100" max-records="0":
    uv run python -m loader load \
        --env {{env}} \
        --batch-size {{batch-size}} \
        --max-records {{max-records}} \
        --no-embed

# Reset checkpoint and reload from scratch
load-reset env="local" batch-size="100" max-records="0":
    uv run python -m loader load \
        --env {{env}} \
        --batch-size {{batch-size}} \
        --max-records {{max-records}} \
        --reset

# Pull the nomic-embed-text model into Ollama (run once before loading)
ollama-pull:
    ollama pull nomic-embed-text

# Check Ollama is running and the embedding model is available
ollama-check:
    ollama list | grep nomic-embed-text || (echo "Model not found — run: just ollama-pull" && exit 1)
    curl -sf http://localhost:11434 > /dev/null && echo "Ollama is reachable" || (echo "Ollama is not running — start with: ollama serve" && exit 1)

# Run all tests
test *args="":
    uv run pytest loader/tests {{args}}

# Run tests with verbose output
test-v *args="":
    uv run pytest loader/tests -v {{args}}

alembic:
    cd db_migrations/
    uv run alembic upgrade head
    cd ..

# Run a vector search against catalog.product_embeddings
# Usage: just search "wireless noise cancelling headphones"
search query env="local" k="10":
    uv run python -m retrieval search "{{query}}" --env {{env}} --k {{k}}

# Run search + lightweight heuristic relevance evaluation
# Usage: just eval "wireless noise cancelling headphones"
eval query env="local" k="10" threshold="0.6":
    uv run python -m retrieval eval "{{query}}" --env {{env}} --k {{k}} --threshold {{threshold}}
