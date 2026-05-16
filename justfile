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

# Pull the default Ollama chat model used by generation/
ollama-pull-llm:
    ollama pull qwen2.5:3b-instruct

# Run a single grounded answer
# Usage: just generate "wireless headphones under \$50"
#        just generate "..." provider=ollama model=qwen2.5:3b-instruct k=5
generate query env="local" provider="openai" model="" k="8" temperature="0.2":
    uv run python -m generation generate '{{query}}' \
        --env {{env}} --provider {{provider}} \
        {{ if model != "" { "--model " + model } else { "" } }} \
        --k {{k}} --temperature {{temperature}}

# Run a single grounded answer using the local Ollama LLM (no OpenAI calls)
# Usage: just generate-local "wireless headphones under \$50"
#        just generate-local "best gaming earbuds" model=llama3.1:8b
#        just generate-local "..." k=5 temperature=0.4
generate-local query env="local" model="qwen2.5:3b-instruct" k="8" temperature="0.2":
    uv run python -m generation generate '{{query}}' \
        --env {{env}} --provider ollama --model {{model}} \
        --k {{k}} --temperature {{temperature}}

# Retrieval + generation + grounding eval
# Usage: just rag-eval "wireless headphones under $50"
#        just rag-eval "..." judge=true
rag-eval query env="local" provider="openai" model="" k="8" threshold="0.6" judge="false":
    uv run python -m generation rag-eval "{{query}}" \
        --env {{env}} --provider {{provider}} \
        {{ if model != "" { "--model " + model } else { "" } }} \
        --k {{k}} --threshold {{threshold}} \
        {{ if judge == "true" { "--judge" } else { "" } }}
