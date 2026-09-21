package usecase_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/000Erick/engram-synapse/internal/domain"
	"github.com/000Erick/engram-synapse/internal/embed"
	"github.com/000Erick/engram-synapse/internal/store"
	"github.com/000Erick/engram-synapse/internal/usecase"
)

// noopReader is a minimal EngramReader for stats tests.
type noopReader struct {
	obs []domain.Observation
	ids []int64
}

func (r *noopReader) LiveObservations(_ context.Context) ([]domain.Observation, error) {
	return r.obs, nil
}

func (r *noopReader) LiveIDs(_ context.Context) ([]int64, error) {
	if r.ids != nil {
		return r.ids, nil
	}
	ids := make([]int64, len(r.obs))
	for i, o := range r.obs {
		ids[i] = o.ID
	}
	return ids, nil
}

func (r *noopReader) FTS(_ context.Context, _ string, _ int) ([]domain.Ranked, error) {
	return nil, nil
}

func newStatsStore(t *testing.T) *store.SQLiteVectorStore {
	t.Helper()
	s, err := store.NewSQLiteVectorStore(filepath.Join(t.TempDir(), "synapse.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestStatsUsecase_ReturnsAllFields(t *testing.T) {
	reader := &noopReader{
		obs: []domain.Observation{
			{ID: 1}, {ID: 2}, {ID: 3}, {ID: 4}, {ID: 5},
		},
	}
	noopEmbed := &embed.NoopEmbedder{}

	uc := usecase.NewStatsUsecase(reader, &store.NoopVectorStore{}, noopEmbed, "text-embedding-3-large")
	result, err := uc.Run(context.Background())
	if err != nil {
		t.Fatalf("stats.Run: %v", err)
	}
	if result.LiveObservations != 5 {
		t.Errorf("expected live_observations=5, got %d", result.LiveObservations)
	}
	if result.Vectors != 0 {
		t.Errorf("expected vectors=0, got %d", result.Vectors)
	}
	if result.Dims != 3072 {
		t.Errorf("expected dims=3072, got %d", result.Dims)
	}
	if result.Model != "text-embedding-3-large" {
		t.Errorf("expected model=text-embedding-3-large, got %q", result.Model)
	}
}

func TestStatsUsecase_ZeroVectors(t *testing.T) {
	reader := &noopReader{}
	noopEmbed := &embed.NoopEmbedder{}

	uc := usecase.NewStatsUsecase(reader, &store.NoopVectorStore{}, noopEmbed, "text-embedding-3-large")
	result, err := uc.Run(context.Background())
	if err != nil {
		t.Fatalf("stats.Run: %v", err)
	}
	if result.LiveObservations != 0 {
		t.Errorf("expected live_observations=0, got %d", result.LiveObservations)
	}
	if result.Orphans != 0 {
		t.Errorf("expected orphans=0, got %d", result.Orphans)
	}
}

// TestStatsUsecase_OrphansComputedCorrectly verifies that orphans are computed
// as "stored IDs not in the live set". Store has {1,2,3}, live has {1,2} → 1 orphan.
func TestStatsUsecase_OrphansComputedCorrectly(t *testing.T) {
	ctx := context.Background()

	reader := &noopReader{
		obs: []domain.Observation{
			{ID: 1, Title: "a", Content: "x"},
			{ID: 2, Title: "b", Content: "y"},
		},
		ids: []int64{1, 2},
	}

	st := newStatsStore(t)
	// Seed vectors for IDs 1, 2, 3 — ID 3 is an orphan (not live).
	rows := []domain.VecRow{
		{ObsID: 1, Embedding: make([]float32, 3072), ContentHash: "h1", Model: "m"},
		{ObsID: 2, Embedding: make([]float32, 3072), ContentHash: "h2", Model: "m"},
		{ObsID: 3, Embedding: make([]float32, 3072), ContentHash: "h3", Model: "m"},
	}
	if err := st.Upsert(ctx, rows); err != nil {
		t.Fatalf("seed: %v", err)
	}

	uc := usecase.NewStatsUsecase(reader, st, &embed.NoopEmbedder{}, "m")
	result, err := uc.Run(ctx)
	if err != nil {
		t.Fatalf("stats.Run: %v", err)
	}
	if result.LiveObservations != 2 {
		t.Errorf("live_observations = %d, want 2", result.LiveObservations)
	}
	if result.Vectors != 3 {
		t.Errorf("vectors = %d, want 3", result.Vectors)
	}
	if result.Orphans != 1 {
		t.Errorf("orphans = %d, want 1 (id=3 is orphan)", result.Orphans)
	}
}
