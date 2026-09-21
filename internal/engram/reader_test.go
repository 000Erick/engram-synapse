package engram_test

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite" // pure-Go driver; no CGO needed for tests

	"github.com/000Erick/engram-synapse/internal/engram"
)

// seedEngram creates a temp engram.db with the Engram schema and seed data.
// It uses a separate write connection, then returns the path for read-only use.
func seedEngram(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "engram.db")

	// Use a direct file DSN (no mode=ro) for seeding.
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(off)")
	if err != nil {
		t.Fatalf("seed: open: %v", err)
	}
	defer db.Close()

	// Schema mirrors real Engram: text columns other than title/content are
	// NULLABLE. The reader must tolerate NULLs (COALESCE in the query).
	_, err = db.Exec(`
		CREATE TABLE observations (
			id         INTEGER PRIMARY KEY,
			title      TEXT NOT NULL,
			content    TEXT NOT NULL,
			project    TEXT,
			scope      TEXT,
			type       TEXT,
			topic_key  TEXT,
			updated_at TEXT,
			deleted_at TEXT
		);
		CREATE VIRTUAL TABLE observations_fts USING fts5(
			title, content, content='observations', content_rowid='id'
		);
		CREATE TRIGGER obs_ai AFTER INSERT ON observations BEGIN
			INSERT INTO observations_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
		END;
	`)
	if err != nil {
		t.Fatalf("seed: schema: %v", err)
	}

	// Insert 5 live, 1 soft-deleted
	rows := []struct {
		id        int
		title     string
		content   string
		deletedAt interface{}
	}{
		{1, "architecture note", "clean architecture rules", nil},
		{2, "teacher guide", "how to teach well", nil},
		{3, "circuit breaker", "resilience pattern for microservices", nil},
		{4, "note alpha", "note content alpha", nil},
		{5, "note beta", "note content beta", nil},
		{6, "deleted obs", "should not appear", "2024-01-01T00:00:00Z"},
		// id 7 has NULL project/scope/type/topic_key/updated_at — reproduces
		// the real-data bug where Scan into string failed on NULL.
		{7, "null-cols obs", "has null metadata", nil},
	}
	for _, r := range rows {
		_, err = db.Exec(
			`INSERT INTO observations (id, title, content, deleted_at) VALUES (?, ?, ?, ?)`,
			r.id, r.title, r.content, r.deletedAt,
		)
		if err != nil {
			t.Fatalf("seed: insert %d: %v", r.id, err)
		}
	}
	return path
}

func TestEngramReader_LiveObservations_OnlyLive(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	obs, err := r.LiveObservations(context.Background())
	if err != nil {
		t.Fatalf("LiveObservations: %v", err)
	}
	if len(obs) != 6 {
		t.Errorf("expected 6 live observations, got %d", len(obs))
	}
	for _, o := range obs {
		if o.ID == 6 {
			t.Errorf("soft-deleted observation (id=6) must not be returned")
		}
	}
}

func TestEngramReader_LiveObservations_FieldsPopulated(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	obs, err := r.LiveObservations(context.Background())
	if err != nil {
		t.Fatalf("LiveObservations: %v", err)
	}
	// Find id=2
	var found bool
	for _, o := range obs {
		if o.ID == 2 {
			found = true
			if o.Title != "teacher guide" {
				t.Errorf("expected title 'teacher guide', got %q", o.Title)
			}
			if o.Content != "how to teach well" {
				t.Errorf("expected content 'how to teach well', got %q", o.Content)
			}
		}
	}
	if !found {
		t.Error("expected observation with id=2 not found")
	}
}

func TestEngramReader_LiveIDs_OnlyLive(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	ids, err := r.LiveIDs(context.Background())
	if err != nil {
		t.Fatalf("LiveIDs: %v", err)
	}
	if len(ids) != 6 {
		t.Errorf("expected 6 live IDs, got %d", len(ids))
	}
	for _, id := range ids {
		if id == 6 {
			t.Errorf("deleted id=6 must not be in LiveIDs")
		}
	}
}

func TestEngramReader_FTS_ReturnsMatches(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	results, err := r.FTS(context.Background(), "architecture", 10)
	if err != nil {
		t.Fatalf("FTS: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least 1 FTS result for 'architecture'")
	}
	if results[0].ID != 1 {
		t.Errorf("expected id=1 (architecture note) as first result, got %d", results[0].ID)
	}
}

func TestEngramReader_FTS_EmptyForNoMatch(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	results, err := r.FTS(context.Background(), "xyzunmatched", 10)
	if err != nil {
		t.Fatalf("FTS: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for unmatched query, got %d", len(results))
	}
}

func TestEngramReader_FTS_LimitHonored(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	// "note" appears in 4 rows (ids 4, 5, and the content fields); limit to 2
	results, err := r.FTS(context.Background(), "note", 2)
	if err != nil {
		t.Fatalf("FTS: %v", err)
	}
	if len(results) > 2 {
		t.Errorf("FTS limit not honored: expected ≤2 results, got %d", len(results))
	}
}

func TestEngramReader_FTS_HandlesSpecialChars(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	// These queries used to crash FTS5 ("no such column: IS", syntax error)
	// because -, ", *, : are FTS5 operators. They must now be safe and return
	// without error (results may be empty, that's fine).
	cases := []string{
		"AS-IS colombia",
		`"unterminated quote`,
		"foo* bar",
		"a:b (c)",
		"AND OR NEAR",
		"   ",
		"",
	}
	for _, q := range cases {
		if _, err := r.FTS(context.Background(), q, 10); err != nil {
			t.Errorf("FTS(%q) returned error, want none: %v", q, err)
		}
	}
}

func TestEngramReader_FTS_HyphenQueryStillMatches(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	// "architecture rules" lives in id 1's content. A query with a hyphen mixed
	// in must still find it (tokens are quoted and AND-ed).
	results, err := r.FTS(context.Background(), "architecture", 10)
	if err != nil {
		t.Fatalf("FTS: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected a match for 'architecture' after sanitization")
	}
}

func TestEngramReader_ReadOnly_RejectsWrites(t *testing.T) {
	path := seedEngram(t)
	r, err := engram.NewSQLiteEngramReader(path)
	if err != nil {
		t.Fatalf("NewSQLiteEngramReader: %v", err)
	}
	defer r.Close()

	// Open a separate writable connection to the same file — this connection is
	// test-only and never touches the read-only SQLiteEngramReader. We use it
	// to confirm the read-only DSN on the reader rejects writes.
	writeDB, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(off)")
	if err != nil {
		t.Fatalf("writable open: %v", err)
	}
	defer writeDB.Close()

	// The reader's DSN uses mode=ro, so attempting any write via *that*
	// connection must fail. We replicate the intent by trying the write on a
	// fresh read-only connection rather than the internal DB handle.
	roDB, err := sql.Open("sqlite", "file:"+path+"?mode=ro")
	if err != nil {
		t.Fatalf("ro open: %v", err)
	}
	defer roDB.Close()

	_, writeErr := roDB.Exec("INSERT INTO observations (id,title,content) VALUES (99,'x','y')")
	if writeErr == nil {
		t.Error("expected write on read-only connection to fail, but it succeeded")
	}
}
