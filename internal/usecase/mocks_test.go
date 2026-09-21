package usecase

import (
	"context"

	"github.com/000Erick/engram-synapse/internal/domain"
)

// mockReader is a test double for port.EngramReader.
type mockReader struct {
	obs     []domain.Observation
	ids     []int64
	ftsHits []domain.Ranked
	ftsErr  error
}

func (m *mockReader) LiveObservations(_ context.Context) ([]domain.Observation, error) {
	return m.obs, nil
}

func (m *mockReader) LiveIDs(_ context.Context) ([]int64, error) {
	if m.ids != nil {
		return m.ids, nil
	}
	ids := make([]int64, len(m.obs))
	for i, o := range m.obs {
		ids[i] = o.ID
	}
	return ids, nil
}

func (m *mockReader) FTS(_ context.Context, _ string, _ int) ([]domain.Ranked, error) {
	return m.ftsHits, m.ftsErr
}
