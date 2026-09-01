# Deploying semblance to local k3s

k3s uses **containerd**, not Docker, so a plain `docker build` is invisible to
the cluster. The flow is: build with Docker, save to a tarball, then import the
tarball into k3s's containerd. The chart sets `imagePullPolicy: Never` so k3s
uses the locally imported image and never contacts a registry.

## Prerequisite

Docker must be reachable from the Ubuntu WSL distro:

```bash
docker version
```

If it says "command not found", enable Docker Desktop → Settings → Resources →
WSL Integration → **Ubuntu-24.04**.

## Build → import → deploy

```bash
cd ~/semblance

# 1. Build the image (multi-stage, static, distroless).
docker build -t semblance:local .

# 2. Save it to a tarball.
docker save semblance:local -o /tmp/semblance-local.tar

# 3. Import into k3s containerd (needs root — this step is yours to run).
sudo k3s ctr images import /tmp/semblance-local.tar

# 4. Install the chart.
helm install semblance ./deploy/helm/semblance

# 5. Verify.
kubectl get pods -l app.kubernetes.io/instance=semblance
kubectl port-forward svc/semblance 8080:8080 &
curl -s localhost:8080/healthz
curl -s localhost:8080/metrics | head
```

Upgrade after a rebuild+reimport: `helm upgrade semblance ./deploy/helm/semblance`
(bump nothing; the pod rolls on config checksum changes).

## Note on the backend and a live chat demo

Inside a pod, `localhost` is the pod, not the WSL host, so the pod cannot reach
Ollama at `localhost:11434`. The deploy proof for this step is the pod Running
plus `/healthz` and `/metrics` answering. For a live chat completion through the
deployed pod, point `config.backendURL` at the WSL host IP (e.g.
`http://<hostIP>:11434/v1`) — that is Step 11 territory.
