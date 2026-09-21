package engram

import (
	"context"

	"github.com/000Erick/engram-synapse/internal/domain"
)

// NoopEngramReader is a graceful-degradation stub used when engram.db is
// unreachable. Every method returns (nil, nil), which downstream usecases treat
// as "no data available" rather than an error — keeping the MCP server alive and
// responsive (with empty results) when Engram is offline.
type NoopEngramReader struct{}

func (n *NoopEngramReader) LiveObservations(_ context.Context) ([]domain.Observation, error) {
	return nil, nil
}

func (n *NoopEngramReader) LiveIDs(_ context.Context) ([]int64, error) {
	return nil, nil
}

func (n *NoopEngramReader) FTS(_ context.Context, _ string, _ int) ([]domain.Ranked, error) {
	return nil, nil
}
