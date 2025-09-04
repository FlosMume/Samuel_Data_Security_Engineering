Databricks Docker Cluster Policies & Container Scanning (CIBC Azure)

This repository contains configured cluster policies and container image scanning setup for running Databricks clusters with Docker images from Azure Container Registry (ACR) in a regulated financial environment (CIBC).

The goal is to enforce security, compliance, and cost control by combining:

Databricks cluster policies (JSON)

Azure DevOps CI/CD pipeline with Trivy scanning

Azure Defender for Containers registry scanning

Networking guardrails (Private Link, VNet Injection, Secure Cluster Connectivity)

📂 Repository Structure
.
├── policies/
│   ├── cluster_policy_prod.json   # PROD jobs clusters – strict, single-user SP, notebooks disabled
│   ├── cluster_policy_dev.json    # DEV all-purpose clusters – relaxed, notebooks allowed
│
├── pipelines/
│   └── azure-pipelines-trivy.yml  # Azure DevOps pipeline: build → scan → push to ACR
│
└── README.md                      # This file

🔐 Cluster Policies

Enforce Docker image allowlists: only images from <acr>.azurecr.io/golden/... (PROD) or <acr>.azurecr.io/sandbox/... (DEV).

Control workloads:

PROD = Jobs-only, Single-user UC mode, notebooks disabled.

DEV = Notebooks allowed, user isolation mode.

Guardrails:

Worker scaling ranges and DBU cost caps.

Approved VM types only.

Local disk encryption enforced.

Logs written to Unity Catalog Volumes path.

Init scripts restricted to approved storage paths.

Tagging: Environment (PROD/DEV) and Cost Center metadata fixed.

Apply these via Databricks UI (Compute → Policies) or using the databricks CLI.

🐳 Container Scanning
1. Build & Scan (CI/CD Gate)

Docker image built and scanned in Azure DevOps pipeline with Trivy.

Pipeline fails on HIGH/CRITICAL CVEs.

Reports stored as SARIF artifacts.

2. Registry Scanning (Azure Defender)

Microsoft Defender for Containers scans images in ACR:

On push/import

On images pulled in last 30 days

Works with Private Link ACR if “Allow trusted Microsoft services” is enabled.

3. Image Governance

Only passing images promoted to golden/ (PROD) or sandbox/ (DEV).

Optional signing/attestation with cosign/Notation.

🌐 Networking & Access

Clusters use VNet Injection + Secure Cluster Connectivity (SCC).

No public IPs on worker nodes.

ACR access via Private Link, public network access disabled.

Docker credentials injected via Databricks secrets.

🚀 Getting Started

Upload cluster policies

databricks cluster-policies create --json-file policies/cluster_policy_prod.json
databricks cluster-policies create --json-file policies/cluster_policy_dev.json


Import pipeline into Azure DevOps

Copy pipelines/azure-pipelines-trivy.yml into your repo.

Configure service connection to your ACR.

Run pipeline to build, scan, and push images.

Enable Defender for Containers

In Azure Portal → Security Center → Defender plans.

Apply at subscription level.

Test new clusters

Create DEV/PROD clusters in Databricks.

Verify:

Docker image pulled from ACR.

Logs appear in Unity Catalog Volumes.

Init script executes from approved location.

📜 License

This repository is provided under the MIT License. Adapt configurations to your own compliance requirements.
