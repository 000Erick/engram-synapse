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
	type pending struct {
		obs  domain.Observation
		hash string
	}
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
		chunk := todo[start:end]

		inputs := make([]string, len(chunk))
		for i, p := range chunk {
			inputs[i] = p.obs.Title + "\n\n" + p.obs.Content
		}

		vecs, err := b.embedder.Embed(ctx, inputs)
		if err != nil {
			res.Failed += int64(len(chunk))
			return res, err
		}
		if len(vecs) != len(chunk) {
			res.Failed += int64(len(chunk))
			return res, errors.New("backfill: embedder returned wrong vector count")
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
			return res, err
		}
		res.Embedded += int64(len(chunk))
	}

	return res, nil
}
