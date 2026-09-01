# syntax=docker/dockerfile:1

# ---- build stage: compile a static, CGO-free binary ----
FROM golang:1.26 AS build
WORKDIR /src

# Cache module downloads separately from the source for faster rebuilds.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# CGO_ENABLED=0 makes a fully static binary that runs on scratch/distroless.
# -trimpath + -ldflags "-s -w" strip paths and debug info to keep it small.
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/semblance ./cmd/semblance

# ---- final stage: distroless static (has CA certs, runs as nonroot) ----
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/semblance /semblance
# The committed price table; SEMBLANCE_PRICE_TABLE points at it.
COPY --from=build /src/config/prices.json /config/prices.json
ENV SEMBLANCE_PRICE_TABLE=/config/prices.json
EXPOSE 8080
ENTRYPOINT ["/semblance"]
