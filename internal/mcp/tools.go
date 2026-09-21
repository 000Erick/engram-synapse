package mcp

import (
	"context"
	"encoding/json"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/000Erick/engram-synapse/internal/port"
	"github.com/000Erick/engram-synapse/internal/usecase"
)

// Deps holds dependencies for MCP tool handlers.
type Deps struct {
	EngramPath  string
	SynapsePath string
	Reader      port.EngramReader
	Store       port.VectorStore
	Embedder    port.Embedder
	Model       string
	APIKey      string
	Version     string
}

// RegisterTools registers all five synapse MCP tools on the server.
func RegisterTools(server *sdkmcp.Server, d *Deps) {
	addHealthTool(server, d)
	addStatsTool(server, d)
	addBackfillTool(server, d)
	addSearchTool(server, d)
	addSyncTool(server, d)
}

// healthArgs has no required parameters.
type healthArgs struct{}

type healthOut struct {
	Status           string `json:"status"`
	EngramReachable  bool   `json:"engram_reachable"`
	SynapseReachable bool   `json:"synapse_reachable"`
	EngramPath       string `json:"engram_path"`
	SynapsePath      string `json:"synapse_path"`
	Version          string `json:"version"`
}

func addHealthTool(server *sdkmcp.Server, d *Deps) {
	sdkmcp.AddTool(server, &sdkmcp.Tool{
		Name:        "synapse_health",
		Description: "Liveness check and DB reachability for both engram.db and synapse.db.",
	}, func(ctx context.Context, _ *sdkmcp.CallToolRequest, _ healthArgs) (*sdkmcp.CallToolResult, any, error) {
		uc := usecase.NewHealthUsecase(d.EngramPath, d.SynapsePath, d.Version)
		r, err := uc.Run(ctx)
		if err != nil {
			return nil, nil, err
		}
		out := healthOut{
			Status:           r.Status,
			EngramReachable:  r.EngramReachable,
			SynapseReachable: r.SynapseReachable,
			EngramPath:       r.EngramPath,
			SynapsePath:      r.SynapsePath,
			Version:          r.Version,
		}
		b, _ := json.Marshal(out)
		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: string(b)}},
		}, nil, nil
	})
}

type statsArgs struct{}

func addStatsTool(server *sdkmcp.Server, d *Deps) {
	sdkmcp.AddTool(server, &sdkmcp.Tool{
		Name:        "synapse_stats",
		Description: "Returns observation count, vector count, orphan count, dims, and model.",
	}, func(ctx context.Context, _ *sdkmcp.CallToolRequest, _ statsArgs) (*sdkmcp.CallToolResult, any, error) {
		result, err := usecase.NewStatsUsecase(d.Reader, d.Store, d.Embedder, d.Model).Run(ctx)
		if err != nil {
			return nil, nil, err
		}
		out := map[string]any{
			"live_observations": result.LiveObservations,
			"vectors":           result.Vectors,
			"orphans":           result.Orphans,
			"dims":              result.Dims,
			"model":             result.Model,
		}
		b, _ := json.Marshal(out)
		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: string(b)}},
		}, nil, nil
	})
}

type backfillArgs struct{}

func addBackfillTool(server *sdkmcp.Server, d *Deps) {
	sdkmcp.AddTool(server, &sdkmcp.Tool{
		Name:        "synapse_backfill",
		Description: "Idempotent batch-embed of all live Engram observations into synapse.db.",
	}, func(ctx context.Context, _ *sdkmcp.CallToolRequest, _ backfillArgs) (*sdkmcp.CallToolResult, any, error) {
		if d.APIKey == "" {
			b, _ := json.Marshal(map[string]string{"error": "OPENAI_API_KEY is not set"})
			return &sdkmcp.CallToolResult{
				IsError: true,
				Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: string(b)}},
			}, nil, nil
		}
		uc := usecase.NewBackfillUsecase(d.Reader, d.Store, d.Embedder, d.APIKey, d.Model)
		result, err := uc.Run(ctx)
		if err != nil {
			return nil, nil, err
		}
		b, _ := json.Marshal(result)
		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: string(b)}},
		}, nil, nil
	})
}

type searchArgs struct {
	Query string `json:"query"`
	Limit int    `json:"limit,omitempty"`
}

func addSearchTool(server *sdkmcp.Server, d *Deps) {
	// Build the search usecase ONCE and share it across all search calls. Its
	// single-flight guard for the background backfill is instance state, so a
	// per-request usecase would let concurrent searches double-embed the same
	// drift. The MCP server is long-lived, so this singleton is safe.
	uc := usecase.NewSelfHealingSearchUsecase(d.Reader, d.Store, d.Embedder, d.APIKey, d.Model)
	sdkmcp.AddTool(server, &sdkmcp.Tool{
		Name:        "synapse_search",
		Description: "Hybrid FTS5+KNN semantic search with RRF fusion. Returns ranked results.",
	}, func(ctx context.Context, _ *sdkmcp.CallToolRequest, args searchArgs) (*sdkmcp.CallToolResult, any, error) {
		// No hard guard here: search degrades gracefully. Without an API key it
		// runs FTS-only; synapse_health is the place to diagnose DB reachability.
		limit := args.Limit
		if limit <= 0 {
			limit = 10
		}
		results, vectorUsed, err := uc.Run(ctx, args.Query, limit)
		if err != nil {
			return nil, nil, err
		}
		out := map[string]any{
			"results":     results,
			"vector_used": vectorUsed,
		}
		b, _ := json.Marshal(out)
		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: string(b)}},
		}, nil, nil
	})
}

type syncArgs struct{}

func addSyncTool(server *sdkmcp.Server, d *Deps) {
	sdkmcp.AddTool(server, &sdkmcp.Tool{
		Name:        "synapse_sync",
		Description: "Detects and removes orphan vectors whose observations were deleted from Engram.",
	}, func(ctx context.Context, _ *sdkmcp.CallToolRequest, _ syncArgs) (*sdkmcp.CallToolResult, any, error) {
		uc := usecase.NewSyncUsecase(d.Reader, d.Store)
		result, err := uc.Run(ctx)
		if err != nil {
			return nil, nil, err
		}
		b, _ := json.Marshal(result)
		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: string(b)}},
		}, nil, nil
	})
}
