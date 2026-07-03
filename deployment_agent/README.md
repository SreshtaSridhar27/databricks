# Deployment Agent (`hon-deployment-agent`)

SDLC worker #4 for the HON Supervisor Agent — a **deterministic deployment pipeline** that
authors artifacts and **safely triggers releases**, running **inside Databricks**. It merges the
other team's Terraform→GitOps *engine* with this project's governed Databricks *shell*.
Full design, node walkthrough, and positives/negatives:
[../Deployment_Agent_Workflow.md](../Deployment_Agent_Workflow.md).

## What it is
A LangGraph **`StateGraph`** (validate → azure_audit → approval → trigger_terraform →
create_gitops_pr → update_jira, with a human-intervention terminal) wrapped in an MLflow
**`ResponsesAgent`**, deployed on **Model Serving** via `agents.deploy()`. GitHub/Jira actions go
through the managed **`system.ai.github` / `system.ai.atlassian` MCP** under **OBO** — no PATs.
Deterministic by design (no free LLM tool loop on prod deploys).

## Files
| File | Purpose |
|------|---------|
| `deployment_agent.py` | The merged agent: `StateGraph` pipeline inside a `ResponsesAgent`, GitHub/Atlassian MCP + OBO, approval gate, Jinja2 rendering. |
| `templates/deployment.yaml.j2` | Externalized K8s Deployment template (edit without touching Python; embedded fallback exists). |
| `log_and_deploy.py` | Log (models-from-code) → register to Unity Catalog → `agents.deploy()`, with `AuthPolicy` (resources + OBO scopes). |
| `deployment_ticket_watcher.py` | **Automated Jira trigger.** Table-update-triggered Job that finds "Ready to Deploy" tickets, verifies the approver, and invokes the agent with structured `custom_inputs`. |
| `requirements.txt` | Runtime deps. |

## Run (Databricks notebook)
```python
%pip install -r requirements.txt
dbutils.library.restartPython()
# then run log_and_deploy.py  → prints the query endpoint
```

## Where things run
Agent brain → **Databricks Model Serving**. `terraform apply` → **GitHub Actions** (triggered).
K8s rollout → your **GitOps controller** after the PR merges. Databricks orchestrates; it does not
run Terraform.

## Safety (two gates)
1. **Code:** staging/prod require the caller in `hon-devops` (OBO) **and** an explicit
   `CONFIRM-DEPLOY-<env>` token; deterministic path, no self-approval.
2. **Platform:** set **MCP Service Policies** to *require approval* / *deny* on deploy tools.

## Before production — resolve the ⚠️ markers
- `LLM_ENDPOINT` + that `ChatDatabricks` accepts `use_ai_gateway=True` on your version.
- Exact `system.ai.github` / `system.ai.atlassian` **external-MCP URLs** and their OAuth scopes;
  discover exact **tool names** with `DatabricksMCPClient(url, ws).list_tools()`
  (`run_workflow`, `create_pull_request`, `create_or_update_file`, `create_branch`, Jira transition/comment).
- Implement `_azure_resource_exists()` (Azure Resource Graph) and the Jira-field → params mapping.
- Add a **status poll/callback** if you must confirm `terraform apply` finished before marking *Deployed*
  (`workflow_dispatch` is async).
- Group surfacing via `current_user.me().groups`; `log_model(name=` vs `artifact_path=` per MLflow version.
- Repos/registry constants (`IAC_REPO`, `GITOPS_REPO`, `CONTAINER_REGISTRY`) → move to `hon_platform.config`.
