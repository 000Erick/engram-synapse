package usecase

import (
	"context"
	"testing"

	"github.com/000Erick/engram-synapse/internal/domain"
)

func seedVec(t *testing.T, st interface {
	Upsert(context.Context, []domain.VecRow) error
}, ids ...int64) {
	t.Helper()
	rows := make([]domain.VecRow, len(ids))
	for i, id := range ids {
		v := make([]float32, 3072)
		v[0] = float32(id)
		rows[i] = domain.VecRow{ObsID: id, Embedding: v, ContentHash: "h", Model: "m"}
	}
	if err := st.Upsert(context.Background(), rows); err != nil {
		t.Fatalf("seed: %v", err)
	}
}

func TestSync_RemovesOrphansKeepsLive(t *testing.T) {
	ctx := context.Background()
	st := newStore(t)
	// Stored vectors: 1,2,3,4,5. Live in engram: 1,3,5 → orphans 2,4.
	seedVec(t, st, 1, 2, 3, 4, 5)
	reader := &mockReader{ids: []int64{1, 3, 5}}

	uc := NewSyncUsecase(reader, st)
	res, err := uc.Run(ctx)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Orphans != 2 {
		t.Errorf("orphans = %d, want 2", res.Orphans)
	}
	if res.LivePreserved != 3 {
		t.Errorf("preserved = %d, want 3", res.LivePreserved)
	}

	remaining, _ := st.AllObsIDs(ctx)
	if len(remaining) != 3 {
		t.Fatalf("remaining = %v, want 3 ids", remaining)
	}
	for _, id := range remaining {
		if id == 2 || id == 4 {
			t.Errorf("orphan %d was not removed", id)
		}
	}
}

func TestSync_SecondRunNoOrphans(t *testing.T) {
	ctx := context.Background()
	st := newStore(t)
	seedVec(t, st, 1, 2, 3)
	reader := &mockReader{ids: []int64{1, 2, 3}}

	uc := NewSyncUsecase(reader, st)
	if _, err := uc.Run(ctx); err != nil {
		t.Fatalf("run 1: %v", err)
	}
	res, err := uc.Run(ctx)
	if err != nil {
		t.Fatalf("run 2: %v", err)
	}
	if res.Orphans != 0 {
		t.Errorf("orphans on clean run = %d, want 0", res.Orphans)
	}
	if res.LivePreserved != 3 {
		t.Errorf("preserved = %d, want 3", res.LivePreserved)
	}
}

func TestSync_AllOrphans(t *testing.T) {
	ctx := context.Background()
	st := newStore(t)
	seedVec(t, st, 10, 11)
	reader := &mockReader{ids: []int64{}} // nothing live

	uc := NewSyncUsecase(reader, st)
	res, err := uc.Run(ctx)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Orphans != 2 {
		t.Errorf("orphans = %d, want 2", res.Orphans)
	}
	n, _ := st.CountVectors(ctx)
	if n != 0 {
		t.Errorf("remaining vectors = %d, want 0", n)
	}
}
