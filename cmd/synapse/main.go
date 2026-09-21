package main

import (
	"context"
	"log"
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/000Erick/engram-synapse/internal/config"
	"github.com/000Erick/engram-synapse/internal/embed"
	"github.com/000Erick/engram-synapse/internal/engram"
	"github.com/000Erick/engram-synapse/internal/mcp"
	"github.com/000Erick/engram-synapse/internal/port"
	"github.com/000Erick/engram-synapse/internal/store"
)

// version is set at build time via ldflags: -X main.version=vX.Y.Z
var version = "dev"

func main() {
	// Load .env for local development. Missing file is not an error — in
	// production config comes from the real environment.
	_ = godotenv.Load()

	cfg, err := config.Load()
	if err != nil {
		log.Printf("synapse: config: %v", err)
		os.Exit(1)
	}

	// deps.Reader is always non-nil so handlers never panic when Engram is
	// missing. The real reader is preferred; the noop stub is used when
	// engram.db is unreachable so the MCP server stays alive with empty results.
	var reader port.EngramReader = &engram.NoopEngramReader{}
	realReader, err := engram.NewSQLiteEngramReader(cfg.EngramDBPath)
	if err != nil {
		log.Printf("synapse: warn: engram not reachable: %v", err)
		// Continue — health tool will report degraded; reader stays noop.
	} else {
		reader = realReader
		defer realReader.Close()
	}

	if err := os.MkdirAll(filepath.Dir(cfg.SynapseDBPath), 0o755); err != nil {
		log.Printf("synapse: warn: cannot create synapse dir: %v", err)
	}
	vecStore, err := store.NewSQLiteVectorStore(cfg.SynapseDBPath)
	if err != nil {
		log.Printf("synapse: vector store: %v", err)
		os.Exit(1)
	}
	defer vecStore.Close()

	// Build the embedder. A custom endpoint and/or dimension let Synapse embed
	// with a free, local OpenAI-compatible model (Ollama, LocalAI, LM Studio)
	// instead of the OpenAI API. Both default to OpenAI when unset.
	var embedOpts []embed.Option
	if cfg.EmbedEndpoint != "" {
		embedOpts = append(embedOpts, embed.WithEndpoint(cfg.EmbedEndpoint))
	}
	if cfg.EmbedDims > 0 {
		embedOpts = append(embedOpts, embed.WithDims(cfg.EmbedDims))
	}
	embedder := embed.NewOpenAIEmbedder(cfg.OpenAIAPIKey, cfg.OpenAIEmbedModel, embedOpts...)

	server := sdkmcp.NewServer(&sdkmcp.Implementation{
		Name:    "synapse",
		Version: version,
	}, nil)

	deps := &mcp.Deps{
		EngramPath:  cfg.EngramDBPath,
		SynapsePath: cfg.SynapseDBPath,
		Reader:      reader,
		Store:       vecStore,
		Embedder:    embedder,
		Model:       cfg.OpenAIEmbedModel,
		APIKey:      cfg.OpenAIAPIKey,
		Version:     version,
	}
	mcp.RegisterTools(server, deps)

	if err := server.Run(context.Background(), &sdkmcp.StdioTransport{}); err != nil {
		log.Printf("synapse: server error: %v", err)
	}
}
