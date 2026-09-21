package store

import (
	"context"
	"database/sql"
	"math"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"

	"github.com/000Erick/engram-synapse/internal/domain"
)

// unitVec builds a Dims-dimensional unit vector by setting component idx to 1
// and all others to zero. This gives cosine-distinct directions for KNN tests.
func unitVec(idx int) []float32 {
	out := make([]float32, Dims)
	out[idx] = 1.0
	return out
}

// mixedVec builds a Dims-dimensional vector with two non-zero components.
// The resulting vector is NOT a unit vector but has a well-defined direction
// useful for testing that cosine ordering is correct.
func mixedVec(a, b float32) []float32 {
	out := make([]float32, Dims)
	out[0] = a
	out[1] = b
	return out
}

// normalise returns a copy of v scaled to unit length.
func normalise(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return v
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out
}

func newTestStore(t *testing.T) *SQLiteVectorStore {
	t.Helper()
	path := filepath.Join(t.TempDir(), "synapse.db")
	s, err := NewSQLiteVectorStore(path)
	if err != nil {
		t.Fatalf("NewSQLiteVectorStore: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// TestVectorStore_KNN_CosineSimilarityOrdering verifies that KNN returns
// results ordered by DESCENDING cosine similarity. The three stored vectors
// each lie in a distinct direction; the query is closest to ID 2.
//
//   - ID 1 → pure-x direction   (axis 0)
//   - ID 2 → mostly-x, tiny-y  (close to query direction)
//   - ID 3 → pure-y direction   (axis 1, perpendicular to ID 1)
//
// Query is pure-x, so expected order: ID 1 ≈ ID 2 > ID 3.
// We use distinct enough angles so ties do not occur in practice.
func TestVectorStore_KNN_CosineSimilarityOrdering(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	// Three vectors with clearly different cosine similarities to query [1,0,...]:
	//   ID 1: [1, 0, ...] → cos = 1.0 (exact match direction)
	//   ID 2: [1, 1, ...] → cos ≈ 0.707 (45° off)
	//   ID 3: [0, 1, ...] → cos = 0.0 (perpendicular)
	rows := []domain.VecRow{
		{ObsID: 1, Embedding: unitVec(0), ContentHash: "h1", Model: "m"}, // cos=1.0
		{ObsID: 2, Embedding: normalise(mixedVec(1, 1)), ContentHash: "h2", Model: "m"}, // cos≈0.707
		{ObsID: 3, Embedding: unitVec(1), ContentHash: "h3", Model: "m"}, // cos=0.0
	}
	if err := s.Upsert(ctx, rows); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	// Query along pure-x axis.
	query := unitVec(0)
	res, err := s.KNN(ctx, query, 3)
	if err != nil {
		t.Fatalf("KNN: %v", err)
	}
	if len(res) != 3 {
		t.Fatalf("expected 3 results, got %d", len(res))
	}
	// Nearest should be ID 1 (cos=1.0).
	if res[0].ID != 1 {
		t.Errorf("nearest = id %d, want id 1 (cos=1.0)", res[0].ID)
	}
	// Second should be ID 2 (cos≈0.707).
	if res[1].ID != 2 {
		t.Errorf("second = id %d, want id 2 (cos≈0.707)", res[1].ID)
	}
	// Third should be ID 3 (cos=0.0).
	if res[2].ID != 3 {
		t.Errorf("third = id %d, want id 3 (cos=0.0)", res[2].ID)
	}
	// Scores must be in descending order (higher cosine similarity = closer).
	if !(res[0].Score >= res[1].Score && res[1].Score >= res[2].Score) {
		t.Errorf("scores not in descending order: %v", res)
	}
	// All results must carry SourceVec.
	for i, r := range res {
		if r.Source != domain.SourceVec {
			t.Errorf("result[%d].Source = %v, want SourceVec", i, r.Source)
		}
	}
}

func TestVectorStore_KNN_HonorsK(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)
	// Insert 5 unit vectors along axis 0..4 — all well-separated in cosine space.
	for i := int64(0); i < 5; i++ {
		if err := s.Upsert(ctx, []domain.VecRow{
			{ObsID: i + 1, Embedding: unitVec(int(i)), ContentHash: "h", Model: "m"},
		}); err != nil {
			t.Fatalf("Upsert %d: %v", i, err)
		}
	}
	res, err := s.KNN(ctx, unitVec(0), 2)
	if err != nil {
		t.Fatalf("KNN: %v", err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %d", len(res))
	}
}

func TestVectorStore_KNN_NearestIsFirst(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	// Three vectors with known cosine similarities to query unitVec(0):
	//   ID 10: cos = 1.0 (identical direction)
	//   ID 20: cos ≈ 0.707
	//   ID 30: cos = 0.0 (perpendicular)
	rows := []domain.VecRow{
		{ObsID: 10, Embedding: unitVec(0), ContentHash: "h10", Model: "m"},
		{ObsID: 20, Embedding: normalise(mixedVec(1, 1)), ContentHash: "h20", Model: "m"},
		{ObsID: 30, Embedding: unitVec(1), ContentHash: "h30", Model: "m"},
	}
	if err := s.Upsert(ctx, rows); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	// k=1 must return only the nearest.
	res, err := s.KNN(ctx, unitVec(0), 1)
	if err != nil {
		t.Fatalf("KNN k=1: %v", err)
	}
	if len(res) != 1 {
		t.Fatalf("k=1: expected 1 result, got %d", len(res))
	}
	if res[0].ID != 10 {
		t.Errorf("k=1: nearest = id %d, want id 10", res[0].ID)
	}
}

func TestVectorStore_Upsert_IsIdempotent(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	row := domain.VecRow{ObsID: 7, Embedding: unitVec(0), ContentHash: "hA", Model: "m"}
	if err := s.Upsert(ctx, []domain.VecRow{row}); err != nil {
		t.Fatalf("Upsert 1: %v", err)
	}
	// Re-upsert same id with a new hash → no duplicate, hash updated.
	row.ContentHash = "hB"
	if err := s.Upsert(ctx, []domain.VecRow{row}); err != nil {
		t.Fatalf("Upsert 2: %v", err)
	}

	n, err := s.CountVectors(ctx)
	if err != nil {
		t.Fatalf("CountVectors: %v", err)
	}
	if n != 1 {
		t.Errorf("count = %d, want 1 (no duplicate)", n)
	}
	hashes, err := s.Hashes(ctx)
	if err != nil {
		t.Fatalf("Hashes: %v", err)
	}
	if hashes[7] != "hB" {
		t.Errorf("hash = %q, want hB (updated)", hashes[7])
	}
}

func TestVectorStore_DeleteByIDs(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	if err := s.Upsert(ctx, []domain.VecRow{
		{ObsID: 1, Embedding: unitVec(0), ContentHash: "h1", Model: "m"},
		{ObsID: 2, Embedding: unitVec(1), ContentHash: "h2", Model: "m"},
	}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := s.DeleteByIDs(ctx, []int64{1}); err != nil {
		t.Fatalf("DeleteByIDs: %v", err)
	}
	n, err := s.CountVectors(ctx)
	if err != nil {
		t.Fatalf("CountVectors: %v", err)
	}
	if n != 1 {
		t.Errorf("count = %d, want 1", n)
	}
	ids, err := s.AllObsIDs(ctx)
	if err != nil {
		t.Fatalf("AllObsIDs: %v", err)
	}
	if len(ids) != 1 || ids[0] != 2 {
		t.Errorf("remaining ids = %v, want [2]", ids)
	}
}

func TestVectorStore_CountOrphans(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	if err := s.Upsert(ctx, []domain.VecRow{
		{ObsID: 1, Embedding: unitVec(0), ContentHash: "h1", Model: "m"},
		{ObsID: 2, Embedding: unitVec(1), ContentHash: "h2", Model: "m"},
		{ObsID: 3, Embedding: unitVec(2), ContentHash: "h3", Model: "m"},
	}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	// Live IDs are 1 and 3 → obs 2 is an orphan.
	orphans, err := s.CountOrphans(ctx, []int64{1, 3})
	if err != nil {
		t.Fatalf("CountOrphans: %v", err)
	}
	if orphans != 1 {
		t.Errorf("orphans = %d, want 1", orphans)
	}
}

// TestNewSQLiteVectorStore_ResetsLegacySchema proves that opening a synapse.db
// whose vec_obs predates the pure-Go schema (here simulated by a table lacking
// the `dims` column) wipes the stale cache and yields a working store.
func TestNewSQLiteVectorStore_ResetsLegacySchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "synapse.db")

	// Create a legacy-shaped synapse.db: vec_obs WITHOUT the `dims` column.
	legacy, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open legacy: %v", err)
	}
	if _, err := legacy.Exec(`CREATE TABLE vec_obs (obs_id INTEGER PRIMARY KEY, embedding BLOB)`); err != nil {
		t.Fatalf("create legacy vec_obs: %v", err)
	}
	if _, err := legacy.Exec(`INSERT INTO vec_obs(obs_id, embedding) VALUES (99, x'00')`); err != nil {
		t.Fatalf("seed legacy row: %v", err)
	}
	legacy.Close()

	// Opening through the constructor must detect the legacy schema, reset the
	// file, and produce a usable store.
	s, err := NewSQLiteVectorStore(path)
	if err != nil {
		t.Fatalf("NewSQLiteVectorStore on legacy db: %v", err)
	}
	t.Cleanup(func() { s.Close() })

	ctx := context.Background()

	// The stale row must be gone (cache was rebuilt from scratch).
	n, err := s.CountVectors(ctx)
	if err != nil {
		t.Fatalf("CountVectors: %v", err)
	}
	if n != 0 {
		t.Fatalf("CountVectors after reset = %d, want 0", n)
	}

	// And the new pure-Go schema works end to end.
	if err := s.Upsert(ctx, []domain.VecRow{
		{ObsID: 1, Embedding: unitVec(0), ContentHash: "h1", Model: "m"},
	}); err != nil {
		t.Fatalf("Upsert after reset: %v", err)
	}
	hits, err := s.KNN(ctx, unitVec(0), 1)
	if err != nil {
		t.Fatalf("KNN after reset: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != 1 {
		t.Fatalf("KNN after reset = %+v, want [{ID:1}]", hits)
	}
}
