package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/000Erick/engram-synapse/internal/port"
)

func noBackoff() Option {
	return WithBackoff(func(int) time.Duration { return 0 })
}

// makeResp builds a valid embeddings JSON response for n inputs, each vector
// having its first element set to the input index for easy assertions.
func writeEmbedResponse(w http.ResponseWriter, n int) {
	type item struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	resp := struct {
		Data []item `json:"data"`
	}{}
	for i := 0; i < n; i++ {
		vec := make([]float32, 3072)
		vec[0] = float32(i)
		resp.Data = append(resp.Data, item{Index: i, Embedding: vec})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func TestOpenAIEmbedder_NoAPIKey(t *testing.T) {
	e := NewOpenAIEmbedder("", "")
	_, err := e.Embed(context.Background(), []string{"hello"})
	if !errors.Is(err, ErrNoAPIKey) {
		t.Fatalf("err = %v, want ErrNoAPIKey", err)
	}
}

func TestOpenAIEmbedder_Success(t *testing.T) {
	var gotInputs []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer k" {
			t.Errorf("auth header = %q", got)
		}
		var req embedRequest
		json.NewDecoder(r.Body).Decode(&req)
		gotInputs = req.Input
		writeEmbedResponse(w, len(req.Input))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", "", WithEndpoint(srv.URL))
	vecs, err := e.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("got %d vectors, want 2", len(vecs))
	}
	if len(vecs[0]) != 3072 {
		t.Errorf("dims = %d, want 3072", len(vecs[0]))
	}
	if vecs[1][0] != 1 {
		t.Errorf("vec[1][0] = %v, want 1 (order preserved)", vecs[1][0])
	}
	if len(gotInputs) != 2 {
		t.Errorf("server saw %d inputs", len(gotInputs))
	}
}

func TestOpenAIEmbedder_RetriesOn429(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n < 3 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		var req embedRequest
		json.NewDecoder(r.Body).Decode(&req)
		writeEmbedResponse(w, len(req.Input))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", "", WithEndpoint(srv.URL), noBackoff())
	vecs, err := e.Embed(context.Background(), []string{"x"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 1 {
		t.Fatalf("got %d vectors", len(vecs))
	}
	if atomic.LoadInt32(&calls) != 3 {
		t.Errorf("calls = %d, want 3 (2 retries)", calls)
	}
}

func TestOpenAIEmbedder_FailFastOn4xx(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"bad"}`))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", "", WithEndpoint(srv.URL), noBackoff())
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error on 400")
	}
	if atomic.LoadInt32(&calls) != 1 {
		t.Errorf("calls = %d, want 1 (no retry on 400)", calls)
	}
}

func TestOpenAIEmbedder_Batches(t *testing.T) {
	var batchSizes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		json.NewDecoder(r.Body).Decode(&req)
		batchSizes = append(batchSizes, len(req.Input))
		writeEmbedResponse(w, len(req.Input))
	}))
	defer srv.Close()

	// 600 inputs → with maxBatch 256 → 256 + 256 + 88 = 3 calls.
	inputs := make([]string, 600)
	for i := range inputs {
		inputs[i] = "t"
	}
	e := NewOpenAIEmbedder("k", "", WithEndpoint(srv.URL), noBackoff())
	vecs, err := e.Embed(context.Background(), inputs)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 600 {
		t.Fatalf("got %d vectors, want 600", len(vecs))
	}
	if len(batchSizes) != 3 {
		t.Fatalf("batches = %v, want 3 calls", batchSizes)
	}
	if batchSizes[0] != 256 || batchSizes[1] != 256 || batchSizes[2] != 88 {
		t.Errorf("batch sizes = %v, want [256 256 88]", batchSizes)
	}
}

func TestOpenAIEmbedder_Dims(t *testing.T) {
	e := NewOpenAIEmbedder("k", "")
	if e.Dims() != 3072 {
		t.Errorf("dims = %d, want 3072", e.Dims())
	}
}

func TestOpenAIEmbedder_WithDims(t *testing.T) {
	e := NewOpenAIEmbedder("k", "", WithDims(768))
	if e.Dims() != 768 {
		t.Errorf("dims = %d, want 768 (local model override)", e.Dims())
	}
}

func TestOpenAIEmbedder_WithDims_IgnoresNonPositive(t *testing.T) {
	e := NewOpenAIEmbedder("k", "", WithDims(0), WithDims(-5))
	if e.Dims() != 3072 {
		t.Errorf("dims = %d, want 3072 (non-positive override ignored)", e.Dims())
	}
}

// A local OpenAI-compatible server (Ollama, LocalAI) is reached purely via the
// endpoint override; the request still carries whatever key is set, which such
// servers ignore. This proves the free/local path works without OpenAI.
func TestOpenAIEmbedder_CustomEndpoint_LocalModel(t *testing.T) {
	var gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		json.NewDecoder(r.Body).Decode(&req)
		gotModel = req.Model
		writeEmbedResponse(w, len(req.Input))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("ollama", "nomic-embed-text", WithEndpoint(srv.URL), WithDims(768))
	vecs, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed against local endpoint: %v", err)
	}
	if len(vecs) != 1 {
		t.Fatalf("got %d vectors, want 1", len(vecs))
	}
	if gotModel != "nomic-embed-text" {
		t.Errorf("server saw model %q, want nomic-embed-text", gotModel)
	}
	if e.Dims() != 768 {
		t.Errorf("dims = %d, want 768", e.Dims())
	}
}

func captureInputs(got *[]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		json.NewDecoder(r.Body).Decode(&req)
		*got = req.Input
		writeEmbedResponse(w, len(req.Input))
	}))
}

func TestOpenAIEmbedder_TruncatesLongInput(t *testing.T) {
	var got []string
	srv := captureInputs(&got)
	defer srv.Close()

	long := strings.Repeat("ñé", 25) // 50 runes, 100 bytes
	inputs := []string{long, "short"}
	e := NewOpenAIEmbedder("k", "", WithEndpoint(srv.URL), WithMaxInputChars(10))
	if _, err := e.Embed(context.Background(), inputs); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("sent %d inputs, want 2", len(got))
	}
	if !utf8.ValidString(got[0]) {
		t.Errorf("truncated input is not valid UTF-8: %q", got[0])
	}
	if want := string([]rune(long)[:10]); got[0] != want {
		t.Errorf("sent %q, want %q", got[0], want)
	}
	if got[1] != "short" {
		t.Errorf("short input = %q, want unchanged", got[1])
	}
	if inputs[0] != long || inputs[1] != "short" {
		t.Error("caller's input slice was mutated")
	}
}

func TestOpenAIEmbedder_DefaultBudget(t *testing.T) {
	var got []string
	srv := captureInputs(&got)
	defer srv.Close()

	e := NewOpenAIEmbedder("k", "", WithEndpoint(srv.URL))
	if _, err := e.Embed(context.Background(), []string{strings.Repeat("é", 40000)}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if n := utf8.RuneCountInString(got[0]); n != 16000 {
		t.Errorf("sent %d runes, want 16000", n)
	}
}

func TestOpenAIEmbedder_WithMaxInputChars_IgnoresNonPositive(t *testing.T) {
	e := NewOpenAIEmbedder("k", "", WithMaxInputChars(0))
	if e.maxInputChars != defaultMaxInputChars {
		t.Errorf("maxInputChars = %d, want %d", e.maxInputChars, defaultMaxInputChars)
	}
}

func TestOpenAIEmbedder_400IsInputRejected(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"maximum input length is 8192 tokens"}`))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", "", WithEndpoint(srv.URL), noBackoff())
	_, err := e.Embed(context.Background(), []string{"x"})
	if !errors.Is(err, port.ErrInputRejected) {
		t.Fatalf("err = %v, want ErrInputRejected", err)
	}
	if c := atomic.LoadInt32(&calls); c != 1 {
		t.Errorf("calls = %d, want 1", c)
	}
}

func TestOpenAIEmbedder_401IsNotInputRejected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("k", "", WithEndpoint(srv.URL), noBackoff())
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil || errors.Is(err, port.ErrInputRejected) {
		t.Fatalf("err = %v, want non-nil and not ErrInputRejected", err)
	}
}
