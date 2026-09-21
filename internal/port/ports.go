package port

import (
	"context"

	"github.com/000Erick/engram-synapse/internal/domain"
)

// EngramReader provides read-only access to Engram observations and FTS5 search.
type EngramReader interface {
	LiveObservations(ctx context.Context) ([]domain.Observation, error)
	LiveIDs(ctx context.Context) ([]int64, error)
	FTS(ctx context.Context, query string, k int) ([]domain.Ranked, error)
}

// Embedder converts text batches into float32 embedding vectors.
type Embedder interface {
	Embed(ctx context.Context, inputs []string) ([][]float32, error)
	Dims() int
}

// VectorStore persists and retrieves embedding vectors.
type VectorStore interface {
	KNN(ctx context.Context, vec []float32, k int) ([]domain.Ranked, error)
	Upsert(ctx context.Context, rows []domain.VecRow) error
	Hashes(ctx context.Context) (map[int64]string, error)
	DeleteByIDs(ctx context.Context, ids []int64) error
	CountVectors(ctx context.Context) (int64, error)
	CountOrphans(ctx context.Context, liveIDs []int64) (int64, error)
	AllObsIDs(ctx context.Context) ([]int64, error)
}
