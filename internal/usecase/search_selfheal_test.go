package usecase

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/000Erick/engram-synapse/internal/domain"
	"github.com/000Erick/engram-synapse/internal/embed"
)

// TestSearch_SyncCure_EmbedsFTSHitMissingVector is the core self-healing test:
// an observation that an agent just saved (present in Engram, matched by FTS,
// but never vectorized) must be embedded inline during the search so it becomes
// rankable on the semantic path in the same call.
func TestSearch_SyncCure_EmbedsFTSHitMissingVector(t *testing.T) {
	ctx := context.Background()

	fresh := domain.Observation{ID: 7, Title: "Fresh", Content: "just saved, no vector yet"}
	reader := &mockReader{
		obs:     []domain.Observation{fresh},
		ftsHits: []domain.Ranked{{ID: 7, Score: 1.0, Source: domain.SourceFTS}},
	}
	st := newStore(t) // empty: obs 7 has no vector
	emb := embedderReturning()

	uc := NewSelfHealingSearchUsecase(reader, st, emb, "key", "text-embedding-3-large")
	res, _, err := uc.Run(ctx, "fresh", 10)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res) == 0 || res[0].ID != 7 {
		t.Fatalf("results = %+v, want hit id=7", res)
	}

	// The synchronous cure must have written a vector for obs 7.
	hashes, err := st.Hashes(ctx)
	if err != nil {
		t.Fatalf("Hashes: %v", err)
	}
	if _, ok := hashes[7]; !ok {
		t.Fatal("obs 7 was not vectorized by the synchronous cure")
	}
}

// TestSearch_SyncCure_SkipsAlreadyVectorized verifies the cure does not re-embed
// observations whose content is unchanged: only the query embedding is sent.
func TestSearch_SyncCure_SkipsAlreadyVectorized(t *testing.T) {
	ctx := context.Background()

	o := domain.Observation{ID: 1, Title: "A", Content: "already vectorized"}
	reader := &mockReader{
		obs:     []domain.Observation{o},
		ftsHits: []domain.Ranked{{ID: 1, Score: 1.0, Source: domain.SourceFTS}},
	}
	st := newStore(t)
	// Seed a vector with the matching content hash so the cure considers it fresh.
	if err := st.Upsert(ctx, []domain.VecRow{{
		ObsID:       1,
		Embedding:   make([]float32, 3072),
		ContentHash: domain.ContentHash(o.Title, o.Content),
		Model:       "m",
	}}); err != nil {
		t.Fatalf("seed: %v", err)
	}

	emb := &embed.MockEmbedder{}
	uc := NewSelfHealingSearchUsecase(reader, st, emb, "key", "m")
	if _, _, err := uc.Run(ctx, "already", 10); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Only the query embedding should have been requested; no cure embed call
	// containing the observation content.
	for _, call := range emb.Snapshot() {
		for _, in := range call {
			if in == o.Title+"\n\n"+o.Content {
				t.Fatal("cure re-embedded an already-vectorized, unchanged observation")
			}
		}
	}
}

// TestSearch_SyncCure_SkippedWithoutAPIKey verifies the cure never runs (and
// never writes vectors) when no API key is configured.
func TestSearch_SyncCure_SkippedWithoutAPIKey(t *testing.T) {
	ctx := context.Background()
	reader := &mockReader{
		obs:     []domain.Observation{{ID: 1, Title: "A", Content: "x"}},
		ftsHits: []domain.Ranked{{ID: 1, Score: 1.0, Source: domain.SourceFTS}},
	}
	st := newStore(t)
	uc := NewSelfHealingSearchUsecase(reader, st, embedderReturning(), "", "m")
	if _, _, err := uc.Run(ctx, "a", 10); err != nil {
		t.Fatalf("Run: %v", err)
	}
	n, _ := st.CountVectors(ctx)
	if n != 0 {
		t.Errorf("vectors = %d, want 0 (no API key → no cure)", n)
	}
}

// TestSearch_BackgroundBackfill_UsesOwnContext proves the async cure does not run
// on the request context (which is cancelled when the search returns). The hook
// records the context the goroutine receives; we assert it is alive after the
// request ctx is cancelled.
func TestSearch_BackgroundBackfill_UsesOwnContext(t *testing.T) {
	reqCtx, cancelReq := context.WithCancel(context.Background())

	reader := &mockReader{
		obs:     []domain.Observation{{ID: 1, Title: "A", Content: "x"}},
		ftsHits: nil,
	}
	uc := NewSelfHealingSearchUsecase(reader, newStore(t), embedderReturning(), "key", "m")

	done := make(chan context.Context, 1)
	uc.newBackgroundCtx = func() (context.Context, context.CancelFunc) {
		bg, cancel := context.WithCancel(context.Background())
		done <- bg
		return bg, cancel
	}

	if _, _, err := uc.Run(reqCtx, "a", 10); err != nil {
		t.Fatalf("Run: %v", err)
	}
	cancelReq() // cancel the request ctx; background must be unaffected

	bg := <-done
	if bg.Err() != nil {
		t.Fatalf("background ctx already done (%v); it must not inherit request cancellation", bg.Err())
	}

	// Let the background goroutine finish before the test ends so it does not
	// race the t.TempDir cleanup writing to synapse.db.
	waitFor(t, func() bool {
		uc.bgMu.Lock()
		defer uc.bgMu.Unlock()
		return !uc.bgRunning
	})
}

// TestSearch_BackgroundBackfill_SingleFlight verifies concurrent searches never
// start two background backfills at once. We block the first background run on a
// gate, fire many concurrent searches while it is in-flight, and assert only one
// backfill started.
func TestSearch_BackgroundBackfill_SingleFlight(t *testing.T) {
	reader := &mockReader{
		obs:     []domain.Observation{{ID: 1, Title: "A", Content: "x"}},
		ftsHits: nil,
	}

	var started int32
	release := make(chan struct{})
	// gatedEmbedder is thread-safe (called from many concurrent searches). It
	// blocks ONLY the background backfill embed call (identified by the
	// observation content title+"\n\n"+content), not the per-query embedding,
	// so the background run stays "in flight" while we fire concurrent searches.
	emb := &gatedEmbedder{
		bgInput: "A" + "\n\n" + "x",
		started: &started,
		release: release,
	}

	uc := NewSelfHealingSearchUsecase(reader, newStore(t), emb, "key", "m")

	// First search kicks the background backfill, which blocks inside Embed.
	if _, _, err := uc.Run(context.Background(), "a", 10); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Wait until the first background run is actually in flight.
	waitFor(t, func() bool { return atomic.LoadInt32(&started) == 1 })

	// Fire many concurrent searches while the first backfill is still blocked.
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, _ = uc.Run(context.Background(), "a", 10)
		}()
	}
	wg.Wait()

	// Single-flight: no second backfill should have started while one was running.
	if got := atomic.LoadInt32(&started); got != 1 {
		t.Fatalf("background backfills started = %d, want 1 (single-flight)", got)
	}

	close(release) // let the held run finish

	// Wait for the background goroutine to finish before the test ends so it does
	// not race the t.TempDir cleanup writing to synapse.db.
	waitFor(t, func() bool {
		uc.bgMu.Lock()
		defer uc.bgMu.Unlock()
		return !uc.bgRunning
	})
}

// gatedEmbedder is a concurrency-safe Embedder for the single-flight test. It
// returns zero-vectors for every input, but blocks (and counts) only the
// background backfill call, identified by its observation input string.
type gatedEmbedder struct {
	bgInput string
	started *int32
	release chan struct{}
}

func (g *gatedEmbedder) Embed(_ context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) == 1 && inputs[0] == g.bgInput {
		atomic.AddInt32(g.started, 1)
		<-g.release // hold the background run open
	}
	out := make([][]float32, len(inputs))
	for i := range out {
		out[i] = make([]float32, 3072)
	}
	return out, nil
}

func (g *gatedEmbedder) Dims() int { return 3072 }

// waitFor spins until cond is true or the test deadline trips. Avoids fixed
// sleeps so the test is fast and not flaky under load.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := make(chan struct{})
	go func() {
		<-t.Context().Done()
		close(deadline)
	}()
	for {
		if cond() {
			return
		}
		select {
		case <-deadline:
			t.Fatal("waitFor: condition not met before deadline")
		default:
			time.Sleep(time.Millisecond)
		}
	}
}
