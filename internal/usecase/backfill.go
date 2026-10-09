package usecase

import (
	"context"
	"errors"

	"github.com/000Erick/engram-synapse/internal/domain"
	"github.com/000Erick/engram-synapse/internal/port"
)

// backfillBatch is how many observations are embedded per OpenAI call / tx.
const backfillBatch = 100

// ErrNoAPIKey is returned when OPENAI_API_KEY is not set.
var ErrNoAPIKey = errors.New("OPENAI_API_KEY is not set")

// BackfillResult holds the counts from a backfill run.
type BackfillResult struct {
	Embedded int64 `json:"embedded"`
	Skipped  int64 `json:"skipped"`
	Failed   int64 `json:"failed"`
	// FailedIDs lists observations the provider rejected even when embedded alone.
	FailedIDs []int64 `json:"failed_ids,omitempty"`
}

// pending is an observation awaiting embedding, with its content hash.
type pending struct {
	obs  domain.Observation
	hash string
}

// BackfillUsecase embeds live observations idempotently.
type BackfillUsecase struct {
	reader    port.EngramReader
	store     port.VectorStore
	embedder  port.Embedder
	apiKey    string
	modelName string
}

// NewBackfillUsecase creates a BackfillUsecase. model is recorded on each stored
// vector for provenance (e.g. "text-embedding-3-large").
func NewBackfillUsecase(reader port.EngramReader, store port.VectorStore, embedder port.Embedder, apiKey, model string) *BackfillUsecase {
	return &BackfillUsecase{reader: reader, store: store, embedder: embedder, apiKey: apiKey, modelName: model}
}

// Run embeds live observations that are new or whose content changed since the
// last run, storing their vectors. Unchanged observations are skipped. The run
// is idempotent: a second run with no changes embeds nothing.
func (b *BackfillUsecase) Run(ctx context.Context) (*BackfillResult, error) {
	if b.apiKey == "" {
		return nil, ErrNoAPIKey
	}

	obs, err := b.reader.LiveObservations(ctx)
	if err != nil {
		return nil, err
	}
	existing, err := b.store.Hashes(ctx)
	if err != nil {
		return nil, err
	}

	res := &BackfillResult{}

	// Collect observations that need embedding.
	var todo []pending
	for _, o := range obs {
		h := domain.ContentHash(o.Title, o.Content)
		if prev, ok := existing[o.ID]; ok && prev == h {
			res.Skipped++
			continue
		}
		todo = append(todo, pending{obs: o, hash: h})
	}

	for start := 0; start < len(todo); start += backfillBatch {
		end := start + backfillBatch
		if end > len(todo) {
			end = len(todo)
		}
		if err := b.embedChunk(ctx, todo[start:end], res); err != nil {
			return res, err
		}
	}

	return res, nil
}

// embedChunk embeds and upserts one chunk. When the provider rejects the
// request content (port.ErrInputRejected) the chunk is bisected: each half is
// embedded recursively, so a single bad input is isolated in O(k·log n) calls
// for k bad inputs among n. A single-item chunk that is still rejected is
// recorded in res.Failed/res.FailedIDs and skipped. Rejected observations are
// not stored, so their hash stays absent and they are retried on the next run.
// Any other error (transient outage, wrong vector count, Upsert failure)
// aborts immediately without extra calls.
func (b *BackfillUsecase) embedChunk(ctx context.Context, chunk []pending, res *BackfillResult) error {
	inputs := make([]string, len(chunk))
	for i, p := range chunk {
		inputs[i] = p.obs.Title + "\n\n" + p.obs.Content
	}

	vecs, err := b.embedder.Embed(ctx, inputs)
	if err != nil {
		if !errors.Is(err, port.ErrInputRejected) {
			res.Failed += int64(len(chunk))
			return err
		}
		if len(chunk) == 1 {
			res.Failed++
			res.FailedIDs = append(res.FailedIDs, chunk[0].obs.ID)
			return nil
		}
		mid := len(chunk) / 2
		if err := b.embedChunk(ctx, chunk[:mid], res); err != nil {
			return err
		}
		return b.embedChunk(ctx, chunk[mid:], res)
	}
	if len(vecs) != len(chunk) {
		res.Failed += int64(len(chunk))
		return errors.New("backfill: embedder returned wrong vector count")
	}

	rows := make([]domain.VecRow, len(chunk))
	for i, p := range chunk {
		rows[i] = domain.VecRow{
			ObsID:       p.obs.ID,
			Embedding:   vecs[i],
			ContentHash: p.hash,
			Model:       b.modelName,
		}
	}
	if err := b.store.Upsert(ctx, rows); err != nil {
		res.Failed += int64(len(chunk))
		return err
	}
	res.Embedded += int64(len(chunk))
	return nil
}
