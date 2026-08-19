// Package openai defines the small subset of the OpenAI chat-completions API
// that the gateway needs to inspect.
//
// Design note on "preserve unknown fields": the gateway never re-serializes a
// client's request from these structs. On a cache miss it forwards the client's
// ORIGINAL request bytes to the backend, and copies the backend's response
// bytes back to the client unchanged. So any field we do not model here — a new
// OpenAI parameter, a provider-specific extension, a client's custom key — still
// reaches the backend untouched. These structs exist only to READ the handful
// of fields we make decisions on (model, stream, params) and to read the token
// counts from the response's `usage` object (the brief: never estimate a token
// count you can read).
package openai

import "encoding/json"

// ChatCompletionRequest is the inspected subset of a POST /v1/chat/completions
// body. Fields we do not list are ignored on decode (Go's default) but are NOT
// lost in transit — see the package doc.
type ChatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`

	// Temperature, TopP and N are pointers so we can distinguish "the client
	// omitted this" (nil) from "the client sent 0" (a real value). This matters
	// for later bypass logic: temperature 0 is a meaningful, cacheable request,
	// whereas an absent temperature should use the backend default. (In Go a
	// plain float64 field can't tell 0 from absent; a *float64 can.)
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	N           *int     `json:"n,omitempty"`

	// Tools and Functions are kept as raw JSON: we only need to know whether
	// they are present (to bypass caching), not to interpret their contents.
	Tools     json.RawMessage `json:"tools,omitempty"`
	Functions json.RawMessage `json:"functions,omitempty"`
}

// Message is one chat turn. Content is deliberately raw JSON because the OpenAI
// schema allows it to be either a plain string or an array of typed parts
// (text, image, etc.); modeling it as a string would fail to decode multimodal
// requests. Helpers that need the text will interpret it when required.
type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// ChatCompletionResponse is the inspected subset of a non-streaming response.
// Used to read Usage for metrics/pricing; the response body itself is passed
// through to the client verbatim rather than rebuilt from this struct.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice is one completion alternative.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage carries the authoritative token counts reported by the backend.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
