package usecase

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/000Erick/engram-synapse/internal/domain"
	"github.com/000Erick/engram-synapse/internal/embed"
	"github.com/000Erick/engram-synapse/internal/store"
)

func newStore(t *testing.T) *store.SQLiteVectorStore {
	t.Helper()
	s, err := store.NewSQLiteVectorStore(filepath.Join(t.TempDir(), "synapse.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// embedderReturning returns vectors whose first element is the input index,
// so each obs gets a distinct deterministic vector.
func embedderReturning() *embed.MockEmbedder {
	return &embed.MockEmbedder{
		EmbedFn: func(inputs []string) ([][]float32, error) {
			out := make([][]float32, len(inputs))
			for i := range inputs {
				v := make([]float32, 3072)
				v[0] = float32(i + 1)
				out[i] = v
			}
			return out, nil
		},
	}
}

func TestBackfill_NoAPIKey(t *testing.T) {
	uc := NewBackfillUsecase(&mockReader{}, newStore(t), &embed.MockEmbedder{}, "", "m")
	_, err := uc.Run(context.Background())
	if !errors.Is(err, ErrNoAPIKey) {
		t.Fatalf("err = %v, want ErrNoAPIKey", err)
	}
}

func TestBackfill_FirstRunEmbedsAll(t *testing.T) {
	reader := &mockReader{obs: []domain.Observation{
		{ID: 1, Title: "a", Content: "x"},
		{ID: 2, Title: "b", Content: "y"},
		{ID: 3, Title: "c", Content: "z"},
	}}
	st := newStore(t)
	uc := NewBackfillUsecase(reader, st, embedderReturning(), "key", "text-embedding-3-large")

	res, err := uc.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Embedded != 3 || res.Skipped != 0 || res.Failed != 0 {
		t.Errorf("first run = %+v, want embedded=3", res)
	}
	n, _ := st.CountVectors(context.Background())
	if n != 3 {
		t.Errorf("stored = %d, want 3", n)
	}
}

func TestBackfill_RerunSkipsUnchanged(t *testing.T) {
	reader := &mockReader{obs: []domain.Observation{
		{ID: 1, Title: "a", Content: "x"},
		{ID: 2, Title: "b", Content: "y"},
	}}
	st := newStore(t)
	emb := embedderReturning()
	uc := NewBackfillUsecase(reader, st, emb, "key", "m")

	if _, err := uc.Run(context.Background()); err != nil {
		t.Fatalf("run 1: %v", err)
	}
	// Second run: nothing changed → 0 embedded, all skipped, no dup.
	res, err := uc.Run(context.Background())
	if err != nil {
		t.Fatalf("run 2: %v", err)
	}
	if res.Embedded != 0 || res.Skipped != 2 {
		t.Errorf("rerun = %+v, want embedded=0 skipped=2", res)
	}
	n, _ := st.CountVectors(context.Background())
	if n != 2 {
		t.Errorf("stored = %d, want 2 (no dupes)", n)
	}
}

func TestBackfill_PartialOnlyChanged(t *testing.T) {
	obs := []domain.Observation{
		{ID: 1, Title: "a", Content: "x"},
		{ID: 2, Title: "b", Content: "y"},
	}
	reader := &mockReader{obs: obs}
	st := newStore(t)
	uc := NewBackfillUsecase(reader, st, embedderReturning(), "key", "m")
	if _, err := uc.Run(context.Background()); err != nil {
		t.Fatalf("run 1: %v", err)
	}

	// Change obs 2's content → only it should re-embed.
	reader.obs[1].Content = "CHANGED"
	res, err := uc.Run(context.Background())
	if err != nil {
		t.Fatalf("run 2: %v", err)
	}
	if res.Embedded != 1 || res.Skipped != 1 {
		t.Errorf("partial = %+v, want embedded=1 skipped=1", res)
	}
}

func TestBackfill_EmbedErrorReported(t *testing.T) {
	reader := &mockReader{obs: []domain.Observation{{ID: 1, Title: "a", Content: "x"}}}
	failing := &embed.MockEmbedder{
		EmbedFn: func(_ []string) ([][]float32, error) {
			return nil, errors.New("boom")
		},
	}
	uc := NewBackfillUsecase(reader, newStore(t), failing, "key", "m")
	res, err := uc.Run(context.Background())
	if err == nil {
		t.Fatal("expected error from embedder")
	}
	if res.Failed != 1 {
		t.Errorf("failed = %d, want 1", res.Failed)
	}
}
