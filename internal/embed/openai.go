package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"time"

	"github.com/000Erick/engram-synapse/internal/port"
)

// ErrNoAPIKey is returned when an embedding is requested but no API key is set.
var ErrNoAPIKey = errors.New("embed: OPENAI_API_KEY is not set")

const (
	defaultEndpoint = "https://api.openai.com/v1/embeddings"
	defaultModel    = "text-embedding-3-large"
	maxBatch        = 256
	maxRetries      = 4

	// defaultMaxInputChars is the per-input rune budget. text-embedding-3-*
	// accept 8192 tokens per input. There is no tokenizer dependency, so we
	// budget in runes conservatively: overflowing 8192 tokens with 16000 runes
	// needs < ~2 chars/token, far below typical prose/markdown/code ratios.
	defaultMaxInputChars = 16000
)

// OpenAIEmbedder calls the OpenAI embeddings API over HTTP.
type OpenAIEmbedder struct {
	apiKey   string
	model    string
	dims     int
	endpoint string
	// maxInputChars is the per-input rune budget; longer inputs are truncated.
	maxInputChars int
	client        *http.Client
	// backoff returns the wait duration before retry attempt n (1-based).
	// Overridable in tests to avoid real sleeps.
	backoff func(attempt int) time.Duration
}

// Option configures an OpenAIEmbedder.
type Option func(*OpenAIEmbedder)

// WithEndpoint overrides the API endpoint (used in tests with httptest).
func WithEndpoint(url string) Option { return func(e *OpenAIEmbedder) { e.endpoint = url } }

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(c *http.Client) Option { return func(e *OpenAIEmbedder) { e.client = c } }

// WithBackoff overrides the retry backoff function (used in tests).
func WithBackoff(f func(attempt int) time.Duration) Option {
	return func(e *OpenAIEmbedder) { e.backoff = f }
}

// WithDims overrides the reported embedding dimensionality. Use it when a
// local OpenAI-compatible model returns vectors of a different size than the
// OpenAI default (e.g. 768 for nomic-embed-text). Non-positive values are
// ignored so the default stays intact. This only affects what Dims() reports;
// the actual stored vector length always follows the API response.
func WithDims(d int) Option {
	return func(e *OpenAIEmbedder) {
		if d > 0 {
			e.dims = d
		}
	}
}

// WithMaxInputChars overrides the per-input rune budget; inputs longer than n
// runes are truncated before being sent. Non-positive values are ignored so
// the default stays intact.
func WithMaxInputChars(n int) Option {
	return func(e *OpenAIEmbedder) {
		if n > 0 {
			e.maxInputChars = n
		}
	}
}

// truncateRunes returns s cut to at most max runes. It counts runes, not
// bytes, and never splits a UTF-8 sequence. s is returned unchanged when it
// fits the budget.
func truncateRunes(s string, max int) string {
	if len(s) <= max {
		return s
	}
	n := 0
	for i := range s {
		if n == max {
			return s[:i]
		}
		n++
	}
	return s
}

const (
	backoffBase    = 250 * time.Millisecond
	backoffMaxWait = 30 * time.Second
)

// defaultBackoff returns an exponential-with-jitter wait for retry attempt n
// (1-based). Formula: base * 2^(n-1) + random fraction of base, capped at 30s.
func defaultBackoff(attempt int) time.Duration {
	exp := time.Duration(math.Pow(2, float64(attempt-1))) * backoffBase
	// Add up to one backoffBase worth of jitter to spread retries.
	jitter := time.Duration(rand.Int63n(int64(backoffBase)))
	d := exp + jitter
	if d > backoffMaxWait {
		d = backoffMaxWait
	}
	return d
}

// NewOpenAIEmbedder builds an embedder. model defaults to text-embedding-3-large
// when empty. The API key may be empty; Embed then returns ErrNoAPIKey.
func NewOpenAIEmbedder(apiKey, model string, opts ...Option) *OpenAIEmbedder {
	if model == "" {
		model = defaultModel
	}
	e := &OpenAIEmbedder{
		apiKey:        apiKey,
		model:         model,
		dims:          defaultDims,
		maxInputChars: defaultMaxInputChars,
		endpoint:      defaultEndpoint,
		client:        &http.Client{Timeout: 60 * time.Second},
		backoff:       defaultBackoff,
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// Dims returns the embedding dimensionality.
func (e *OpenAIEmbedder) Dims() int { return e.dims }

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed returns one vector per input, preserving order. Inputs are split into
// batches of at most maxBatch. Returns ErrNoAPIKey if no key is configured.
func (e *OpenAIEmbedder) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	if e.apiKey == "" {
		return nil, ErrNoAPIKey
	}
	if len(inputs) == 0 {
		return [][]float32{}, nil
	}

	// Truncate on a copy so the caller's slice is never mutated.
	trimmed := make([]string, len(inputs))
	for i, in := range inputs {
		trimmed[i] = truncateRunes(in, e.maxInputChars)
	}
	inputs = trimmed

	out := make([][]float32, 0, len(inputs))
	for start := 0; start < len(inputs); start += maxBatch {
		end := start + maxBatch
		if end > len(inputs) {
			end = len(inputs)
		}
		vecs, err := e.embedBatch(ctx, inputs[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (e *OpenAIEmbedder) embedBatch(ctx context.Context, batch []string) ([][]float32, error) {
	body, err := json.Marshal(embedRequest{Model: e.model, Input: batch})
	if err != nil {
		return nil, fmt.Errorf("embed: marshal: %w", err)
	}

	var lastErr error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("embed: new request: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := e.client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("embed: http: %w", err)
			// network error → retry with backoff
			if attempt < maxRetries {
				if err := sleep(ctx, e.backoff(attempt)); err != nil {
					return nil, err
				}
				continue
			}
			return nil, lastErr
		}

		switch {
		case resp.StatusCode == http.StatusOK:
			vecs, parseErr := parseEmbedResponse(resp.Body, len(batch))
			resp.Body.Close()
			return vecs, parseErr
		case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
			// transient → retry
			drain(resp.Body)
			lastErr = fmt.Errorf("embed: status %d", resp.StatusCode)
			if attempt < maxRetries {
				if err := sleep(ctx, e.backoff(attempt)); err != nil {
					return nil, err
				}
				continue
			}
			return nil, lastErr
		default:
			// 4xx other than 429 → fail fast
			msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
			resp.Body.Close()
			switch resp.StatusCode {
			case http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
				return nil, fmt.Errorf("embed: status %d: %s: %w", resp.StatusCode, string(msg), port.ErrInputRejected)
			}
			return nil, fmt.Errorf("embed: status %d: %s", resp.StatusCode, string(msg))
		}
	}
	return nil, lastErr
}

func parseEmbedResponse(r io.Reader, want int) ([][]float32, error) {
	var er embedResponse
	if err := json.NewDecoder(r).Decode(&er); err != nil {
		return nil, fmt.Errorf("embed: decode: %w", err)
	}
	if len(er.Data) != want {
		return nil, fmt.Errorf("embed: expected %d vectors, got %d", want, len(er.Data))
	}
	out := make([][]float32, want)
	for _, d := range er.Data {
		if d.Index < 0 || d.Index >= want {
			return nil, fmt.Errorf("embed: bad index %d", d.Index)
		}
		out[d.Index] = d.Embedding
	}
	return out, nil
}

func drain(r io.ReadCloser) {
	io.Copy(io.Discard, io.LimitReader(r, 4096))
	r.Close()
}

func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
