package store

import (
	"container/heap"
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	_ "modernc.org/sqlite" // registers the "sqlite" driver; pure Go, no CGO

	"github.com/000Erick/engram-synapse/internal/domain"
)

// SQLiteVectorStore is a modernc.org/sqlite backed implementation of
// port.VectorStore. It owns synapse.db and never touches engram.db.
type SQLiteVectorStore struct {
	db *sql.DB
}

// NewSQLiteVectorStore opens (or creates) synapse.db at path and ensures the
// vec_obs table and companion obs_meta table exist. WAL mode and a 5-second
// busy timeout are set using modernc.org/sqlite URI pragma syntax so
// concurrent readers and the background backfill goroutine do not starve each
// other.
func NewSQLiteVectorStore(path string) (*SQLiteVectorStore, error) {
	// A synapse.db created by the old sqlite-vec (CGO) backend stores vectors in
	// a `vec0` virtual table the pure-Go driver cannot read. synapse.db is a
	// derived cache, so we just delete and rebuild it; synapse_backfill (or the
	// self-healing search) repopulates it on the next run.
	if err := resetIfLegacy(path); err != nil {
		return nil, err
	}
	// modernc.org/sqlite uses URI query params for pragmas. Use the
	// _pragma=<name>(<value>) form — this is the only supported syntax.
	dsn := path + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("store: open: %w", err)
	}
	if err := db.PingContext(context.Background()); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: ping: %w", err)
	}
	if _, err := db.Exec(schemaVecObs); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: create vec_obs: %w", err)
	}
	if _, err := db.Exec(schemaObsMeta); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: create obs_meta: %w", err)
	}
	return &SQLiteVectorStore{db: db}, nil
}

// Close closes the underlying DB connection.
func (s *SQLiteVectorStore) Close() error {
	return s.db.Close()
}

// resetIfLegacy deletes synapse.db (and its WAL/SHM sidecars) when its vec_obs
// table was created by the old sqlite-vec backend, or by any schema that lacks
// the current `dims` column. Detection reads sqlite_master.sql (a plain table,
// safe to query without the vec0 module); the delete is done at the filesystem
// level so we never have to DROP a virtual table whose C module is unavailable.
func resetIfLegacy(path string) error {
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return nil // fresh install, nothing to migrate
	}

	db, err := sql.Open("sqlite", path+"?mode=ro")
	if err != nil {
		return fmt.Errorf("store: inspect legacy db: %w", err)
	}
	var ddl string
	scanErr := db.QueryRow(
		`SELECT COALESCE(sql, '') FROM sqlite_master WHERE type = 'table' AND name = 'vec_obs'`,
	).Scan(&ddl)
	db.Close()

	if errors.Is(scanErr, sql.ErrNoRows) {
		return nil // no vec_obs yet; CREATE TABLE will make a fresh one
	}
	if scanErr != nil {
		return fmt.Errorf("store: inspect legacy db: %w", scanErr)
	}

	low := strings.ToLower(ddl)
	legacy := strings.Contains(low, "virtual") || strings.Contains(low, "vec0") || !strings.Contains(low, "dims")
	if !legacy {
		return nil // already the pure-Go schema
	}

	for _, p := range []string{path, path + "-wal", path + "-shm"} {
		if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("store: reset legacy db %q: %w", p, err)
		}
	}
	return nil
}

// encodeVec serializes a float32 slice as a little-endian byte sequence.
func encodeVec(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

// decodeVec deserializes a little-endian byte sequence into a float32 slice.
func decodeVec(b []byte) []float32 {
	n := len(b) / 4
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

// cosineSimilarity returns the cosine similarity between two float32 vectors.
// Both inputs are assumed to have the same length. Returns 0 if either vector
// has zero norm (guard against divide-by-zero). OpenAI embeddings are unit-
// normalised so the result is stable in practice.
func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		fa, fb := float64(a[i]), float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// --- bounded min-heap for KNN -----------------------------------------------

// knnItem is a single candidate in the bounded heap.
type knnItem struct {
	id    int64
	score float64
}

// knnHeap is a min-heap ordered by score (lowest score at the top). We keep
// the k HIGHEST scores, so when the heap exceeds capacity we pop the minimum.
type knnHeap []knnItem

func (h knnHeap) Len() int            { return len(h) }
func (h knnHeap) Less(i, j int) bool  { return h[i].score < h[j].score } // min at top
func (h knnHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *knnHeap) Push(x interface{}) { *h = append(*h, x.(knnItem)) }
func (h *knnHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// KNN returns up to k nearest neighbours to vec ordered by descending cosine
// similarity (higher = closer). The result Ranked.Score is the cosine
// similarity value.
//
// Implementation: streaming brute-force with a bounded min-heap of size k.
// Rows are scanned one at a time; peak memory usage is proportional to k (not
// the total number of stored vectors). Rows whose stored dims differ from the
// query vector length are silently skipped (defensive guard; should never
// happen in a well-formed store).
func (s *SQLiteVectorStore) KNN(ctx context.Context, vec []float32, k int) ([]domain.Ranked, error) {
	if k <= 0 {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, `SELECT obs_id, embedding, dims FROM vec_obs`)
	if err != nil {
		return nil, fmt.Errorf("store: KNN scan: %w", err)
	}
	defer rows.Close()

	h := make(knnHeap, 0, k+1)
	heap.Init(&h)

	queryLen := len(vec)

	for rows.Next() {
		var (
			id   int64
			blob []byte
			dims int
		)
		if err := rows.Scan(&id, &blob, &dims); err != nil {
			return nil, fmt.Errorf("store: KNN row scan: %w", err)
		}
		// Defensive: skip rows whose stored dimensionality doesn't match query.
		if dims != queryLen {
			continue
		}
		stored := decodeVec(blob)
		score := cosineSimilarity(vec, stored)

		heap.Push(&h, knnItem{id: id, score: score})
		if h.Len() > k {
			heap.Pop(&h) // evict the lowest-scoring item
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: KNN rows: %w", err)
	}

	// Extract results from the heap and sort descending by score.
	result := make([]domain.Ranked, h.Len())
	for i := len(result) - 1; i >= 0; i-- {
		item := heap.Pop(&h).(knnItem)
		result[i] = domain.Ranked{ID: item.id, Score: item.score, Source: domain.SourceVec}
	}
	return result, nil
}

// Upsert inserts or replaces vectors and their metadata in a single
// transaction. Re-upserting the same obs_id overwrites the prior vector/hash.
func (s *SQLiteVectorStore) Upsert(ctx context.Context, rows []domain.VecRow) error {
	if len(rows) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("store: begin: %w", err)
	}
	defer tx.Rollback()

	for _, r := range rows {
		blob := encodeVec(r.Embedding)
		dims := len(r.Embedding)

		// Normal table supports UPSERT syntax directly; no need for delete+insert.
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO vec_obs(obs_id, embedding, dims) VALUES(?, ?, ?)
             ON CONFLICT(obs_id) DO UPDATE SET embedding=excluded.embedding, dims=excluded.dims`,
			r.ObsID, blob, dims,
		); err != nil {
			return fmt.Errorf("store: upsert vec %d: %w", r.ObsID, err)
		}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO obs_meta(obs_id, content_hash, model, created_at)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(obs_id) DO UPDATE SET content_hash = excluded.content_hash, model = excluded.model`,
			r.ObsID, r.ContentHash, r.Model, time.Now().UTC().Format(time.RFC3339),
		); err != nil {
			return fmt.Errorf("store: upsert meta %d: %w", r.ObsID, err)
		}
	}
	return tx.Commit()
}

// Hashes returns a map of obs_id -> content_hash for all stored vectors, used
// by backfill to skip unchanged observations.
func (s *SQLiteVectorStore) Hashes(ctx context.Context) (map[int64]string, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT obs_id, content_hash FROM obs_meta`)
	if err != nil {
		return nil, fmt.Errorf("store: Hashes: %w", err)
	}
	defer rows.Close()

	out := make(map[int64]string)
	for rows.Next() {
		var (
			id int64
			h  string
		)
		if err := rows.Scan(&id, &h); err != nil {
			return nil, fmt.Errorf("store: Hashes scan: %w", err)
		}
		out[id] = h
	}
	return out, rows.Err()
}

// DeleteByIDs removes vectors and metadata for the given obs_ids in one tx.
func (s *SQLiteVectorStore) DeleteByIDs(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("store: begin: %w", err)
	}
	defer tx.Rollback()

	for _, id := range ids {
		if _, err := tx.ExecContext(ctx, `DELETE FROM vec_obs WHERE obs_id = ?`, id); err != nil {
			return fmt.Errorf("store: del vec %d: %w", id, err)
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM obs_meta WHERE obs_id = ?`, id); err != nil {
			return fmt.Errorf("store: del meta %d: %w", id, err)
		}
	}
	return tx.Commit()
}

// CountVectors returns the number of stored vectors.
func (s *SQLiteVectorStore) CountVectors(ctx context.Context) (int64, error) {
	var n int64
	if err := s.db.QueryRowContext(ctx, `SELECT count(*) FROM obs_meta`).Scan(&n); err != nil {
		return 0, fmt.Errorf("store: CountVectors: %w", err)
	}
	return n, nil
}

// CountOrphans returns how many stored vectors have an obs_id that is NOT in
// the given set of live IDs.
func (s *SQLiteVectorStore) CountOrphans(ctx context.Context, liveIDs []int64) (int64, error) {
	all, err := s.AllObsIDs(ctx)
	if err != nil {
		return 0, err
	}
	live := make(map[int64]struct{}, len(liveIDs))
	for _, id := range liveIDs {
		live[id] = struct{}{}
	}
	var orphans int64
	for _, id := range all {
		if _, ok := live[id]; !ok {
			orphans++
		}
	}
	return orphans, nil
}

// AllObsIDs returns every obs_id currently stored.
func (s *SQLiteVectorStore) AllObsIDs(ctx context.Context) ([]int64, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT obs_id FROM obs_meta ORDER BY obs_id`)
	if err != nil {
		return nil, fmt.Errorf("store: AllObsIDs: %w", err)
	}
	defer rows.Close()

	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("store: AllObsIDs scan: %w", err)
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}
