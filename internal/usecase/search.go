package usecase

import (
	"context"
	"errors"
	"sync"
	"unicode/utf8"

	"github.com/000Erick/engram-synapse/internal/domain"
	"github.com/000Erick/engram-synapse/internal/port"
)

const (
	rrfK            = 60
	defaultLimit    = 10
	candidateFactor = 5 // pull this many * limit candidates from each source
)

// SearchUsecase performs hybrid FTS5+KNN search with RRF fusion.
//
// It is self-healing: it does not depend on any agent remembering to call
// synapse_backfill after a mem_save. On every search it cures stale vectors in
// two speeds:
//
//  1. Synchronous (priority): observations that matched FTS but have no vector
//     yet are embedded inline, so the result the caller sees already includes
//     freshly-saved memories on the semantic path.
//  2. Asynchronous (best-effort): a full-drift backfill runs in the background
//     on its own context, so observations saved by other agents that did not
//     trigger backfill converge to a 100% vectorized index for the next search.
//
// The background backfill is single-flighted so concurrent searches never run
// two full backfills at once (which would double-embed the same drift).
type SearchUsecase struct {
	reader    port.EngramReader
	store     port.VectorStore
	embedder  port.Embedder
	apiKey    string
	modelName string

	// bgBackfill performs the asynchronous full-drift cure. May be nil in tests
	// or when no API key is configured, in which case the async cure is skipped.
	bgBackfill *BackfillUsecase

	// bgMu guards bgRunning to single-flight the background backfill.
	bgMu      sync.Mutex
	bgRunning bool

	// newBackgroundCtx builds the context for the async backfill. It defaults to
	// context.Background so the goroutine survives the request ctx cancellation;
	// tests override it to assert isolation and to await completion.
	newBackgroundCtx func() (context.Context, context.CancelFunc)
}

// NewSearchUsecase creates a SearchUsecase with self-healing disabled (no model
// name, no background backfill). Kept for tests and callers that only need pure
// hybrid search.
func NewSearchUsecase(reader port.EngramReader, store port.VectorStore, embedder port.Embedder, apiKey string) *SearchUsecase {
	return &SearchUsecase{
		reader:           reader,
		store:            store,
		embedder:         embedder,
		apiKey:           apiKey,
		newBackgroundCtx: func() (context.Context, context.CancelFunc) { return context.Background(), func() {} },
	}
}

// NewSelfHealingSearchUsecase creates a SearchUsecase that cures stale vectors
// on every search: synchronously for FTS hits and asynchronously for the rest
// of the drift. model is recorded on vectors written by the synchronous cure.
func NewSelfHealingSearchUsecase(reader port.EngramReader, store port.VectorStore, embedder port.Embedder, apiKey, model string) *SearchUsecase {
	uc := NewSearchUsecase(reader, store, embedder, apiKey)
	uc.modelName = model
	if apiKey != "" {
		uc.bgBackfill = NewBackfillUsecase(reader, store, embedder, apiKey, model)
	}
	return uc
}

// Run executes a hybrid search. It runs FTS5 (lexical) and vector KNN (semantic)
// concurrently, fuses them with RRF (k=60), and hydrates the top results from
// Engram. The bool return reports whether the vector path was used (false when
// it was unavailable and the result is FTS-only).
func (s *SearchUsecase) Run(ctx context.Context, query string, limit int) ([]domain.SearchResult, bool, error) {
	if query == "" {
		return nil, false, errors.New("query must not be empty")
	}
	if limit <= 0 {
		limit = defaultLimit
	}
	candidates := limit * candidateFactor

	// FTS path runs first so its hits can drive the synchronous cure before the
	// vector path queries KNN. FTS reads engram.db live, so it always sees
	// freshly-saved observations even when they have no vector yet.
	ftsHits, ftsErr := s.reader.FTS(ctx, query, candidates)
	if ftsErr != nil {
		return nil, false, ftsErr
	}

	// Fetch live observations ONCE — reused by cureFTSHits and hydrate so we
	// only read engram.db a single time per search request.
	obs, err := s.reader.LiveObservations(ctx)
	if err != nil {
		return nil, false, err
	}
	byID := make(map[int64]domain.Observation, len(obs))
	for _, o := range obs {
		byID[o.ID] = o
	}

	// Synchronous cure (priority: the agent searching now). Embed FTS hits that
	// have no vector yet so the vector path can rank them in this same search.
	s.cureFTSHitsWithMap(ctx, ftsHits, byID)

	var vecHits []domain.Ranked
	var vectorUsed bool

	// The query embed runs after FTS + synchronous cure (not concurrently): the
	// cure must finish before KNN so freshly-saved observations rank in this same
	// search. Added latency is one local engram.db read plus any cure embeds.
	// Vector path (only when an API key is configured; otherwise FTS-only).
	if s.apiKey != "" {
		vecs, err := s.embedder.Embed(ctx, []string{query})
		if err == nil && len(vecs) > 0 {
			if hits, err := s.store.KNN(ctx, vecs[0], candidates); err == nil {
				vecHits = hits
				vectorUsed = true
			}
			// KNN failure is non-fatal: degrade to FTS-only rather than erroring.
		}
		// Embed failure is non-fatal: degrade to FTS-only rather than erroring.
	}

	// Asynchronous cure (best-effort: drift left by other agents). Fire-and-forget
	// so it never blocks the response; runs on its own context, single-flighted.
	s.kickBackgroundBackfill()

	lists := [][]domain.Ranked{ftsHits}
	if vectorUsed {
		lists = append(lists, vecHits)
	}
	fused := domain.RRF(lists, rrfK)
	if len(fused) > limit {
		fused = fused[:limit]
	}

	results := hydrateFromMap(fused, byID)
	return results, vectorUsed, nil
}

// hydrateFromMap turns ranked IDs into SearchResults using the pre-fetched
// observation map. The map is built once per search in Run so engram.db is
// only read a single time.
func hydrateFromMap(ranked []domain.Ranked, byID map[int64]domain.Observation) []domain.SearchResult {
	if len(ranked) == 0 {
		return []domain.SearchResult{}
	}
	out := make([]domain.SearchResult, 0, len(ranked))
	for _, r := range ranked {
		o, ok := byID[r.ID]
		if !ok {
			continue // ranked id no longer live; skip
		}
		out = append(out, domain.SearchResult{
			ID:      o.ID,
			Title:   o.Title,
			Snippet: snippet(o.Content, 200),
			Score:   r.Score,
			Source:  r.Source,
		})
	}
	return out
}

func snippet(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:n])
}

// cureFTSHitsWithMap embeds, synchronously, the FTS-matched observations that
// have no vector yet (or whose content changed). It accepts the pre-built byID
// map from Run so LiveObservations is not called a second time per search.
// This is the priority path: it keeps added latency bounded to the handful of
// hits the caller is actually searching for. Best-effort: any failure degrades
// to FTS-only silently.
func (s *SearchUsecase) cureFTSHitsWithMap(ctx context.Context, ftsHits []domain.Ranked, byID map[int64]domain.Observation) {
	if s.apiKey == "" || len(ftsHits) == 0 {
		return
	}

	existing, err := s.store.Hashes(ctx)
	if err != nil {
		return // best-effort
	}

	type pending struct {
		obs  domain.Observation
		hash string
	}
	var todo []pending
	for _, h := range ftsHits {
		o, ok := byID[h.ID]
		if !ok {
			continue // ranked id no longer live
		}
		ch := domain.ContentHash(o.Title, o.Content)
		if prev, ok := existing[o.ID]; ok && prev == ch {
			continue // already vectorized and unchanged
		}
		todo = append(todo, pending{obs: o, hash: ch})
	}
	if len(todo) == 0 {
		return
	}

	inputs := make([]string, len(todo))
	for i, p := range todo {
		inputs[i] = p.obs.Title + "\n\n" + p.obs.Content
	}
	vecs, err := s.embedder.Embed(ctx, inputs)
	if err != nil || len(vecs) != len(todo) {
		return // best-effort
	}
	rows := make([]domain.VecRow, len(todo))
	for i, p := range todo {
		rows[i] = domain.VecRow{
			ObsID:       p.obs.ID,
			Embedding:   vecs[i],
			ContentHash: p.hash,
			Model:       s.modelName,
		}
	}
	_ = s.store.Upsert(ctx, rows) // best-effort
}

// kickBackgroundBackfill launches a full-drift backfill in the background unless
// one is already running. It runs on its own context (not the request ctx, which
// is cancelled when the search returns) so the goroutine survives. Single-flight
// ensures concurrent searches never start two backfills that would double-embed
// the same drift. When there is no drift, BackfillUsecase.Run embeds nothing and
// makes no OpenAI call, so kicking it on every search is effectively free.
func (s *SearchUsecase) kickBackgroundBackfill() {
	if s.bgBackfill == nil {
		return
	}
	s.bgMu.Lock()
	if s.bgRunning {
		s.bgMu.Unlock()
		return
	}
	s.bgRunning = true
	s.bgMu.Unlock()

	go func() {
		ctx, cancel := s.newBackgroundCtx()
		defer cancel()
		defer func() {
			s.bgMu.Lock()
			s.bgRunning = false
			s.bgMu.Unlock()
		}()
		_, _ = s.bgBackfill.Run(ctx) // best-effort
	}()
}
