package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OpenAI is an Embedder backed by the OpenAI embeddings HTTP API. It speaks the
// wire protocol directly (no SDK), per the build brief's stdlib-only rule. Used
// only in the live demo; tests use Fake.
type OpenAI struct {
	baseURL string // e.g. https://api.openai.com/v1
	apiKey  string
	model   string // e.g. text-embedding-3-small
	dim     int    // e.g. 1536
	client  *http.Client
}

// NewOpenAI builds an OpenAI embedder. baseURL should include the /v1 segment.
func NewOpenAI(baseURL, apiKey, model string, dim int, client *http.Client) *OpenAI {
	if client == nil {
		client = &http.Client{}
	}
	return &OpenAI{baseURL: baseURL, apiKey: apiKey, model: model, dim: dim, client: client}
}

func (o *OpenAI) Dimensions() int { return o.dim }

type embeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (o *OpenAI) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(embeddingRequest{Model: o.model, Input: text})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode != http.StatusOK {
		// Surface a trimmed body so demo failures are diagnosable, without
		// dumping an unbounded response into an error string.
		return nil, fmt.Errorf("embeddings API status %d: %s", resp.StatusCode, trim(raw, 256))
	}

	var out embeddingResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decode embeddings response: %w", err)
	}
	if len(out.Data) == 0 || len(out.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("embeddings API returned no embedding")
	}
	return out.Data[0].Embedding, nil
}

func trim(b []byte, n int) string {
	if len(b) > n {
		return string(b[:n]) + "…"
	}
	return string(b)
}
