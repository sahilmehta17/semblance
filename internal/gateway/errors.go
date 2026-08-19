package gateway

import (
	"encoding/json"
	"net/http"
)

// OpenAI error responses have the shape {"error": {"message","type","param","code"}}.
// Clients and SDKs parse this envelope, so our errors must match it rather than
// returning a bare string.
type errorEnvelope struct {
	Error errorBody `json:"error"`
}

type errorBody struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"` // always null for us, but the field must exist
	Code    *string `json:"code"`
}

// writeError sends an OpenAI-shaped error with the given HTTP status.
func writeError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	// Encoding a fixed, small struct cannot realistically fail; ignore the error
	// rather than pretend to handle it.
	_ = json.NewEncoder(w).Encode(errorEnvelope{
		Error: errorBody{Message: message, Type: errType},
	})
}

// logAndWriteError logs the underlying detail (which may include internal URLs)
// and returns a generic message to the client, so we never leak backend
// topology in a response body. It logs through the request-scoped logger so the
// line carries the request ID.
func (s *Server) logAndWriteError(w http.ResponseWriter, r *http.Request, status int, errType, clientMsg string, detail error) {
	s.reqLogger(r).Error("request failed", "type", errType, "status", status, "err", detail)
	writeError(w, status, errType, clientMsg)
}
