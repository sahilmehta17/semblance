{{- define "semblance.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "semblance.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "semblance.labels" -}}
app.kubernetes.io/name: {{ include "semblance.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "semblance.selectorLabels" -}}
app.kubernetes.io/name: {{ include "semblance.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
