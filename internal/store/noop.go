package store

import (
	"context"

	"github.com/000Erick/engram-synapse/internal/domain"
)

// NoopVectorStore is a stub that satisfies the VectorStore port without
// any real storage. Used as a placeholder when no real store is needed.
type NoopVectorStore struct{}

func (n *NoopVectorStore) KNN(_ context.Context, _ []float32, _ int) ([]domain.Ranked, error) {
	return nil, nil
}

func (n *NoopVectorStore) Upsert(_ context.Context, _ []domain.VecRow) error {
	return nil
}

func (n *NoopVectorStore) Hashes(_ context.Context) (map[int64]string, error) {
	return map[int64]string{}, nil
}

func (n *NoopVectorStore) DeleteByIDs(_ context.Context, _ []int64) error {
	return nil
}

func (n *NoopVectorStore) CountVectors(_ context.Context) (int64, error) {
	return 0, nil
}

func (n *NoopVectorStore) CountOrphans(_ context.Context, _ []int64) (int64, error) {
	return 0, nil
}

func (n *NoopVectorStore) AllObsIDs(_ context.Context) ([]int64, error) {
	return nil, nil
}
