package domain_test

import (
	"testing"

	"github.com/000Erick/engram-synapse/internal/domain"
)

func TestRRF_TwoLists_FusedOrder(t *testing.T) {
	// Spec scenario: FTS=[A,B,C] KNN=[B,A,D] → B and A tied at top, C and D below
	// RRF k=60
	// B: 1/62 + 1/61 ≈ 0.03255
	// A: 1/61 + 1/62 ≈ 0.03255
	// C: 1/63 ≈ 0.01587
	// D: 1/63 ≈ 0.01587
	fts := []domain.Ranked{
		{ID: 1, Score: 1, Source: domain.SourceFTS}, // A
		{ID: 2, Score: 2, Source: domain.SourceFTS}, // B
		{ID: 3, Score: 3, Source: domain.SourceFTS}, // C
	}
	knn := []domain.Ranked{
		{ID: 2, Score: 1, Source: domain.SourceVec}, // B
		{ID: 1, Score: 2, Source: domain.SourceVec}, // A
		{ID: 4, Score: 3, Source: domain.SourceVec}, // D
	}

	result := domain.RRF([][]domain.Ranked{fts, knn}, 60)

	if len(result) != 4 {
		t.Fatalf("expected 4 results, got %d", len(result))
	}

	// B and A are tied; both must appear before C and D
	top2IDs := map[int64]bool{result[0].ID: true, result[1].ID: true}
	if !top2IDs[1] || !top2IDs[2] {
		t.Errorf("expected A(1) and B(2) in top 2, got ids %d and %d", result[0].ID, result[1].ID)
	}
	// C and D are below
	bottom2IDs := map[int64]bool{result[2].ID: true, result[3].ID: true}
	if !bottom2IDs[3] || !bottom2IDs[4] {
		t.Errorf("expected C(3) and D(4) in bottom 2, got ids %d and %d", result[2].ID, result[3].ID)
	}
}

func TestRRF_BothSources_SourceBitmask(t *testing.T) {
	// Observation appearing in both lists gets Source = SourceBoth
	fts := []domain.Ranked{{ID: 10, Score: 1, Source: domain.SourceFTS}}
	knn := []domain.Ranked{{ID: 10, Score: 1, Source: domain.SourceVec}}

	result := domain.RRF([][]domain.Ranked{fts, knn}, 60)

	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}
	if result[0].Source != domain.SourceBoth {
		t.Errorf("expected Source=SourceBoth(%d), got %d", domain.SourceBoth, result[0].Source)
	}
}

func TestRRF_SingleList_ScoreCorrect(t *testing.T) {
	// Observation E only in FTS at rank 1: score = 1/(60+1+1) = 1/62... wait
	// rank is 0-based, so rank=0 → 1/(60+0+1)=1/61
	fts := []domain.Ranked{{ID: 5, Score: 1, Source: domain.SourceFTS}}

	result := domain.RRF([][]domain.Ranked{fts}, 60)

	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}
	expected := 1.0 / float64(60+0+1) // rank 0-based
	if abs(result[0].Score-expected) > 1e-9 {
		t.Errorf("expected score %.9f, got %.9f", expected, result[0].Score)
	}
}

func TestRRF_EmptyLists_ReturnsEmpty(t *testing.T) {
	result := domain.RRF([][]domain.Ranked{nil, nil}, 60)
	if len(result) != 0 {
		t.Errorf("expected 0 results for empty lists, got %d", len(result))
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
