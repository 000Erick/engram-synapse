package engram_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/000Erick/engram-synapse/internal/embed"
	"github.com/000Erick/engram-synapse/internal/engram"
	"github.com/000Erick/engram-synapse/internal/store"
	"github.com/000Erick/engram-synapse/internal/usecase"
)

// TestNoopEngramReader_BackfillDoesNotPanic verifies that synapse_backfill
// does not panic when constructed with a NoopEngramReader (i.e., when engram.db
// is unreachable at startup).
func TestNoopEngramReader_BackfillDoesNotPanic(t *testing.T) {
	ctx := context.Background()
	reader := &engram.NoopEngramReader{}

	st, err := store.NewSQLiteVectorStore(filepath.Join(t.TempDir(), "synapse.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	defer st.Close()

	emb := &embed.NoopEmbedder{}
	uc := usecase.NewBackfillUsecase(reader, st, emb, "key", "m")
	result, err := uc.Run(ctx)
	if err != nil {
		t.Fatalf("backfill with noop reader: %v", err)
	}
	// Noop reader returns nil obs → nothing to embed.
	if result.Embedded != 0 {
		t.Errorf("embedded = %d, want 0 (no observations)", result.Embedded)
	}
}

// TestNoopEngramReader_SyncDoesNotPanic verifies that synapse_sync does not
// panic when constructed with a NoopEngramReader.
func TestNoopEngramReader_SyncDoesNotPanic(t *testing.T) {
	ctx := context.Background()
	reader := &engram.NoopEngramReader{}

	st, err := store.NewSQLiteVectorStore(filepath.Join(t.TempDir(), "synapse.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	defer st.Close()

	uc := usecase.NewSyncUsecase(reader, st)
	result, err := uc.Run(ctx)
	if err != nil {
		t.Fatalf("sync with noop reader: %v", err)
	}
	if result.Orphans != 0 {
		t.Errorf("orphans = %d, want 0 (nothing stored)", result.Orphans)
	}
}
