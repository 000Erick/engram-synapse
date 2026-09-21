// Command smoke runs a real backfill + search against the live Engram DB.
// It is a manual tool, not part of the test suite. Requires OPENAI_API_KEY.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/joho/godotenv"

	"github.com/000Erick/engram-synapse/internal/config"
	"github.com/000Erick/engram-synapse/internal/embed"
	"github.com/000Erick/engram-synapse/internal/engram"
	"github.com/000Erick/engram-synapse/internal/store"
	"github.com/000Erick/engram-synapse/internal/usecase"
)

func main() {
	_ = godotenv.Load()
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	if cfg.OpenAIAPIKey == "" {
		log.Fatal("OPENAI_API_KEY not set")
	}

	ctx := context.Background()

	reader, err := engram.NewSQLiteEngramReader(cfg.EngramDBPath)
	if err != nil {
		log.Fatalf("engram: %v", err)
	}
	defer reader.Close()

	if err := os.MkdirAll(filepath.Dir(cfg.SynapseDBPath), 0o755); err != nil {
		log.Fatalf("mkdir: %v", err)
	}
	vstore, err := store.NewSQLiteVectorStore(cfg.SynapseDBPath)
	if err != nil {
		log.Fatalf("store: %v", err)
	}
	defer vstore.Close()

	embedder := embed.NewOpenAIEmbedder(cfg.OpenAIAPIKey, cfg.OpenAIEmbedModel)

	// Backfill only embeds new/changed observations (idempotent), so this is
	// cheap when already populated.
	bf := usecase.NewBackfillUsecase(reader, vstore, embedder, cfg.OpenAIAPIKey, cfg.OpenAIEmbedModel)
	res, err := bf.Run(ctx)
	if err != nil {
		log.Fatalf("backfill: %v", err)
	}
	fmt.Printf("backfill: embedded=%d skipped=%d failed=%d\n", res.Embedded, res.Skipped, res.Failed)

	queries := os.Args[1:]
	if len(queries) == 0 {
		queries = []string{"colombia"}
	}
	search := usecase.NewSearchUsecase(reader, vstore, embedder, cfg.OpenAIAPIKey)
	for _, q := range queries {
		fmt.Printf("\n== search %q ==\n", q)
		hits, vectorUsed, err := search.Run(ctx, q, 8)
		if err != nil {
			fmt.Printf("  error: %v\n", err)
			continue
		}
		fmt.Printf("  vectorUsed=%v, %d hits:\n", vectorUsed, len(hits))
		for i, h := range hits {
			src := map[int]string{1: "fts", 2: "vec", 3: "both"}[int(h.Source)]
			fmt.Printf("  %d. [%-4s score=%.4f] #%d %s\n", i+1, src, h.Score, h.ID, h.Title)
		}
	}
}
