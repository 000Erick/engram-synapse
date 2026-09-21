package usecase

import (
	"context"

	"github.com/000Erick/engram-synapse/internal/port"
)

// SyncResult holds the counts from a sync run.
type SyncResult struct {
	Orphans       int64 `json:"removed"`
	LivePreserved int64 `json:"live_preserved"`
}

// SyncUsecase removes orphan vectors.
type SyncUsecase struct {
	reader port.EngramReader
	store  port.VectorStore
}

// NewSyncUsecase creates a SyncUsecase.
func NewSyncUsecase(reader port.EngramReader, store port.VectorStore) *SyncUsecase {
	return &SyncUsecase{reader: reader, store: store}
}

// Run removes vectors whose observation id is no longer live in Engram (either
// hard-deleted or soft-deleted via deleted_at). Live vectors are preserved.
func (s *SyncUsecase) Run(ctx context.Context) (*SyncResult, error) {
	liveIDs, err := s.reader.LiveIDs(ctx)
	if err != nil {
		return nil, err
	}
	storedIDs, err := s.store.AllObsIDs(ctx)
	if err != nil {
		return nil, err
	}

	live := make(map[int64]struct{}, len(liveIDs))
	for _, id := range liveIDs {
		live[id] = struct{}{}
	}

	var orphans []int64
	var preserved int64
	for _, id := range storedIDs {
		if _, ok := live[id]; ok {
			preserved++
			continue
		}
		orphans = append(orphans, id)
	}

	if len(orphans) > 0 {
		if err := s.store.DeleteByIDs(ctx, orphans); err != nil {
			return nil, err
		}
	}

	return &SyncResult{Orphans: int64(len(orphans)), LivePreserved: preserved}, nil
}
