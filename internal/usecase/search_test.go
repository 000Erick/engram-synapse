package usecase

import (
	"context"
	"testing"

	"github.com/000Erick/engram-synapse/internal/domain"
	"github.com/000Erick/engram-synapse/internal/embed"
)

func TestSearch_EmptyQueryErrors(t *testing.T) {
	uc := NewSearchUsecase(&mockReader{}, newStore(t), &embed.MockEmbedder{}, "key")
	_, _, err := uc.Run(context.Background(), "", 10)
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestSearch_NoAPIKey_FTSOnly(t *testing.T) {
	// Without an API key, only FTS runs. Reader returns one FTS hit.
	reader := &mockReader{
		obs:     []domain.Observation{{ID: 1, Title: "Doc", Content: "hello world"}},
		ftsHits: []domain.Ranked{{ID: 1, Score: 2.0, Source: domain.SourceFTS}},
	}
	uc := NewSearchUsecase(reader, newStore(t), &embed.MockEmbedder{}, "")
	res, vectorUsed, err := uc.Run(context.Background(), "hello", 10)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if vectorUsed {
		t.Error("vectorUsed = true, want false (no API key)")
	}
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("results = %+v, want one hit id=1", res)
	}
	if res[0].Source != domain.SourceFTS {
		t.Errorf("source = %d, want FTS", res[0].Source)
	}
}

// TestSearch_CrossLanguage is the acceptance test: a Spanish query ("profesor")
// must surface an English observation ("teacher") via the vector path, even when
// FTS5 finds nothing (no shared lexical token).
func TestSearch_CrossLanguage_ProfesorFindsTeacher(t *testing.T) {
	ctx := context.Background()

	// English observation about a teacher.
	teacher := domain.Observation{ID: 42, Title: "Teacher dashboard", Content: "Project built by a teacher"}
	reader := &mockReader{
		obs:     []domain.Observation{teacher},
		ftsHits: nil, // FTS finds nothing for "profesor"
	}

	st := newStore(t)

	// Vector for "teacher" content. We store it, then make the embedder return a
	// near-identical vector for the query "profesor" so KNN ranks it first.
	teacherVec := make([]float32, 3072)
	teacherVec[0] = 1.0
	teacherVec[1] = 0.5
	if err := st.Upsert(ctx, []domain.VecRow{
		{ObsID: 42, Embedding: teacherVec, ContentHash: "h", Model: "m"},
	}); err != nil {
		t.Fatalf("seed vector: %v", err)
	}

	// Embedder maps the Spanish query to a vector close to the teacher vector.
	emb := &embed.MockEmbedder{
		EmbedFn: func(inputs []string) ([][]float32, error) {
			q := make([]float32, 3072)
			q[0] = 1.0
			q[1] = 0.49 // very close to teacherVec → nearest neighbour
			return [][]float32{q}, nil
		},
	}

	uc := NewSearchUsecase(reader, st, emb, "key")
	res, vectorUsed, err := uc.Run(ctx, "profesor", 10)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !vectorUsed {
		t.Fatal("vectorUsed = false, want true (semantic path)")
	}
	if len(res) == 0 {
		t.Fatal("no results — Spanish query failed to surface English observation")
	}
	if res[0].ID != 42 {
		t.Errorf("top result id = %d, want 42 (teacher)", res[0].ID)
	}
	if res[0].Source&domain.SourceVec == 0 {
		t.Errorf("source = %d, want vector bit set", res[0].Source)
	}
}

func TestSearch_RRFFusion_BothSources(t *testing.T) {
	ctx := context.Background()

	obs := []domain.Observation{
		{ID: 1, Title: "A", Content: "alpha"},
		{ID: 2, Title: "B", Content: "beta"},
	}
	reader := &mockReader{
		obs:     obs,
		ftsHits: []domain.Ranked{{ID: 1, Score: 1, Source: domain.SourceFTS}},
	}
	st := newStore(t)
	v := make([]float32, 3072)
	v[0] = 1
	if err := st.Upsert(ctx, []domain.VecRow{{ObsID: 1, Embedding: v, ContentHash: "h", Model: "m"}}); err != nil {
		t.Fatalf("seed: %v", err)
	}
	emb := &embed.MockEmbedder{EmbedFn: func(_ []string) ([][]float32, error) {
		q := make([]float32, 3072)
		q[0] = 1
		return [][]float32{q}, nil
	}}

	uc := NewSearchUsecase(reader, st, emb, "key")
	res, _, err := uc.Run(ctx, "alpha", 10)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// Obs 1 found by both FTS and vector → source should be SourceBoth.
	var found bool
	for _, r := range res {
		if r.ID == 1 {
			found = true
			if r.Source != domain.SourceBoth {
				t.Errorf("obs 1 source = %d, want SourceBoth(%d)", r.Source, domain.SourceBoth)
			}
		}
	}
	if !found {
		t.Error("obs 1 missing from fused results")
	}
}

// TestSnippet_MultibyteUTF8 verifies that snippet() does not split a multibyte
// UTF-8 character. Both accented Latin and CJK characters are exercised.
func TestSnippet_MultibyteUTF8(t *testing.T) {
	cases := []struct {
		input string
		n     int
		want  string
	}{
		// 201 runes all from ASCII — plain truncation at 200 runes.
		{"a" + string(make([]rune, 200)) + "b", 200, "a" + string(make([]rune, 199))},
		// Accented Latin: each char is 2 bytes.  Truncating to 3 runes must stay valid UTF-8.
		{"áéíóú", 3, "áéí"},
		// CJK: each char is 3 bytes. Truncate to 2 → "你好".
		{"你好世界", 2, "你好"},
		// Short string shorter than n — returned as-is.
		{"café", 100, "café"},
	}
	for _, tc := range cases {
		got := snippet(tc.input, tc.n)
		if got != tc.want {
			t.Errorf("snippet(%q, %d) = %q, want %q", tc.input, tc.n, got, tc.want)
		}
	}
}

func TestSearch_LimitHonored(t *testing.T) {
	ctx := context.Background()
	obs := make([]domain.Observation, 20)
	hits := make([]domain.Ranked, 20)
	for i := range obs {
		id := int64(i + 1)
		obs[i] = domain.Observation{ID: id, Title: "t", Content: "c"}
		hits[i] = domain.Ranked{ID: id, Score: float64(20 - i), Source: domain.SourceFTS}
	}
	reader := &mockReader{obs: obs, ftsHits: hits}
	uc := NewSearchUsecase(reader, newStore(t), &embed.MockEmbedder{}, "")
	res, _, err := uc.Run(ctx, "t", 5)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res) != 5 {
		t.Errorf("got %d results, want 5", len(res))
	}
}
