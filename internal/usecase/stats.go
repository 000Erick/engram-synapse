package usecase

import (
	"context"
	"fmt"

	"github.com/000Erick/engram-synapse/internal/port"
)

// StatsResult is the output of StatsUsecase.Run.
type StatsResult struct {
	LiveObservations int64
	Vectors          int64
	Orphans          int64
	Dims             int
	Model            string
}

// StatsUsecase collects statistics from both databases.
type StatsUsecase struct {
	reader   port.EngramReader
	store    port.VectorStore
	embedder port.Embedder
	model    string
}

// NewStatsUsecase creates a new StatsUsecase. reader is used to count live
// observations and to obtain the live ID set for the orphan calculation.
func NewStatsUsecase(reader port.EngramReader, store port.VectorStore, embedder port.Embedder, model string) *StatsUsecase {
	return &StatsUsecase{reader: reader, store: store, embedder: embedder, model: model}
}

// Run fetches statistics from both engram.db (via reader) and synapse.db (via
// store). Orphans are correctly computed by diffing live IDs against stored IDs.
func (s *StatsUsecase) Run(ctx context.Context) (*StatsResult, error) {
	obs, err := s.reader.LiveObservations(ctx)
	if err != nil {
		return nil, fmt.Errorf("stats: LiveObservations: %w", err)
	}

	liveIDs, err := s.reader.LiveIDs(ctx)
	if err != nil {
		return nil, fmt.Errorf("stats: LiveIDs: %w", err)
	}

	vectors, err := s.store.CountVectors(ctx)
	if err != nil {
		return nil, fmt.Errorf("stats: CountVectors: %w", err)
	}

	orphans, err := s.store.CountOrphans(ctx, liveIDs)
	if err != nil {
		return nil, fmt.Errorf("stats: CountOrphans: %w", err)
	}

	return &StatsResult{
		LiveObservations: int64(len(obs)),
		Vectors:          vectors,
		Orphans:          orphans,
		Dims:             s.embedder.Dims(),
		Model:            s.model,
	}, nil
}
