"""
Deployment Agent — SDLC worker #4 (hon-deployment-agent) for the HON Supervisor Agent.

MERGED BEST SOLUTION (both teams)
---------------------------------
* Deployment ENGINE  (from the other team's "DA" POC): a deterministic 5-stage pipeline —
  validate -> azure_audit -> (approval) -> trigger_terraform -> create_gitops_pr -> update_jira,
  with conditional routing (skip Terraform if the resource already exists) and a
  human-intervention terminal. Kept as-is because deployment must be DETERMINISTIC, not a
  free-wheeling LLM tool loop.
* Governed SHELL  (this project): the pipeline is a LangGraph `StateGraph` wrapped in an MLflow
  `ResponsesAgent`, deployed on Model Serving via `agents.deploy()` and called by the supervisor
  or the Jira watcher. It runs INSIDE Databricks with OBO identity, Unity Catalog RBAC, AI Gateway
  guardrails, MLflow tracing, and an explicit approval gate.

WHAT CHANGED vs the standalone POC (so it runs governed inside Databricks):
  - GitHub PATs in .env         -> managed `system.ai.github` MCP + OBO (short-lived OAuth)
  - local jira_payload.json     -> request `custom_inputs` (supervisor) or Jira (Atlassian MCP)
  - raw requests.post()         -> MCP tool calls (same workflow_dispatch, now audited)
  - no governance               -> UC + OBO + AI Gateway + MLflow tracing + approval gate

WHERE THINGS RUN
  - Agent brain (this graph): Databricks Model Serving (fast, in-process orchestration).
  - `terraform apply`: GitHub Actions runners, triggered via workflow_dispatch (never blocks here).
  - K8s reconcile: your GitOps controller after the PR merges.

⚠️ = confirm the exact value/URL/tool-name/scope against the live docs on HON's workspace.
Discover exact MCP tool names once with `DatabricksMCPClient(url, ws).list_tools()`.
"""

from __future__ import annotations

import json
from typing import Generator, Optional, TypedDict

import mlflow
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import ModelServingUserCredentials
from databricks_langchain import ChatDatabricks
from databricks_mcp import DatabricksMCPClient
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.graph import END, StateGraph

mlflow.langchain.autolog()  # end-to-end tracing of every node/tool call (audit + debugging)

# ----------------------------------------------------------------------------------
# Config  (move to hon_platform.config Delta tables / env vars in production)
# ----------------------------------------------------------------------------------
LLM_ENDPOINT = "hon-ai-gateway-llm"          # ⚠️ AI Gateway-fronted Claude endpoint (used for summaries)
AUTHORIZED_DEPLOY_GROUP = "hon-devops"       # only this group may deploy to staging/prod
GATED_ENVIRONMENTS = {"staging", "prod"}     # environments that require human approval
VALID_ENVIRONMENTS = {"dev", "staging", "prod"}
REQUIRED_FIELDS = ("resource_name", "resource_type", "environment")

# Service principals allowed on the AUTOMATED (Jira-driven) path. They still must supply a valid
# approval token, which the watcher mints only AFTER verifying (Jira changelog) that a DevOps user
# approved the ticket. See deployment_ticket_watcher.py.
SERVICE_APPROVER_PRINCIPALS = {"hon-deployment-watcher-sp"}  # ⚠️ set to the watcher job's SP

# Repos / registry (⚠️ from hon_platform.config in production)
IAC_REPO = "honeywell/iac"                   # Terraform repo
GITOPS_REPO = "honeywell/gitops"             # K8s manifests repo
TERRAFORM_WORKFLOW = "terraform-deploy.yml"  # workflow_dispatch target in IAC_REPO
CONTAINER_REGISTRY = "honeywell.azurecr.io"

# resource_type -> Terraform module (the other team's mapping)
MODULE_FOR = {"redis": "redis", "postgres": "postgres", "storage": "storage"}

# Managed MCP tool names (⚠️ confirm via list_tools(); GitHub server preserves old names as aliases)
GH_RUN_WORKFLOW = "run_workflow"             # GitHub Actions workflow_dispatch
GH_CREATE_BRANCH = "create_branch"
GH_PUT_FILE = "create_or_update_file"
GH_CREATE_PR = "create_pull_request"
JIRA_TRANSITION = "transition_issue"         # ⚠️ Atlassian MCP tool names vary
JIRA_COMMENT = "add_comment"
JIRA_GET = "get_issue"

EMBEDDED_K8S_TEMPLATE = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ resource_name }}
  namespace: {{ environment }}
spec:
  replicas: {{ replicas }}
  selector: { matchLabels: { app: {{ resource_name }} } }
  template:
    metadata: { labels: { app: {{ resource_name }} } }
    spec:
      containers:
        - name: {{ resource_name }}
          image: {{ container_registry }}/{{ resource_name }}:{{ version }}
          resources:
            limits: { cpu: "{{ cpu_limit }}", memory: "{{ memory_limit }}" }
"""


# ----------------------------------------------------------------------------------
# URL + client helpers
# ----------------------------------------------------------------------------------
def _github_mcp_url(host: str) -> str:
    return f"{host}/api/2.0/mcp/external/system/ai/github"      # ⚠️ verify external-MCP path

def _atlassian_mcp_url(host: str) -> str:
    return f"{host}/api/2.0/mcp/external/system/ai/atlassian"   # ⚠️ verify external-MCP path

def _obo_workspace_client() -> WorkspaceClient:
    """Per-request client acting AS THE END USER (Model Serving OBO). Build in the request path only."""
    return WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

def _current_user_and_groups(ws: WorkspaceClient) -> tuple[str, list[str]]:
    me = ws.current_user.me()
    groups = [g.display for g in (me.groups or []) if g.display]  # ⚠️ verify SCIM group surfacing
    return (me.user_name or "unknown", groups)

def _mcp(ws: WorkspaceClient, url: str) -> DatabricksMCPClient:
    return DatabricksMCPClient(server_url=url, workspace_client=ws)


# ----------------------------------------------------------------------------------
# Jinja2 K8s rendering (externalized template + embedded fallback — the other team's pattern)
# ----------------------------------------------------------------------------------
def _render_manifest(params: dict) -> str:
    try:
        env = Environment(loader=FileSystemLoader("templates"),
                          autoescape=select_autoescape(enabled_extensions=()))
        tmpl = env.get_template("deployment.yaml.j2")
    except Exception:
        tmpl = Environment(autoescape=False).from_string(EMBEDDED_K8S_TEMPLATE)
    return tmpl.render(
        resource_name=params["resource_name"],
        environment=params["environment"],
        replicas=params.get("replicas", 1),
        cpu_limit=params.get("cpu_limit", "250m"),
        memory_limit=params.get("memory_limit", "256Mi"),
        container_registry=CONTAINER_REGISTRY,
        resource_type=params["resource_type"],
        version=params.get("version", "latest"),
    )


# ----------------------------------------------------------------------------------
# Pipeline state
# ----------------------------------------------------------------------------------
class DeployState(TypedDict, total=False):
    ticket_key: str
    resource_name: str
    resource_type: str
    environment: str
    version: str
    replicas: int
    cpu_limit: str
    memory_limit: str
    approval_confirmation: str
    actor: str
    authorized: bool          # actor in hon-devops or an allowlisted service principal
    resource_exists: bool
    tf_dispatched: bool
    pr_url: str
    jira_updated: bool
    status: str               # DEPLOYED | PENDING_APPROVAL | VALIDATION_FAILED | DEPLOY_FAILED
    error: str
    log: list


# ----------------------------------------------------------------------------------
# Graph builder — nodes close over the per-request OBO clients
# ----------------------------------------------------------------------------------
def _build_graph(user_ws: WorkspaceClient):
    host = user_ws.config.host
    gh = _mcp(user_ws, _github_mcp_url(host))
    jira = _mcp(user_ws, _atlassian_mcp_url(host))

    def _log(state: DeployState, msg: str) -> None:
        state.setdefault("log", []).append(msg)

    # Node 0 — optionally hydrate params from the Jira deployment ticket
    def load_ticket(state: DeployState) -> DeployState:
        if state.get("ticket_key") and not state.get("resource_name"):
            try:
                issue = jira.call_tool(JIRA_GET, {"issueIdOrKey": state["ticket_key"]})
                # ⚠️ map your Jira fields -> params here (fields live under issue.content/JSON)
                _log(state, f"Loaded ticket {state['ticket_key']} from Jira.")
            except Exception as e:  # noqa: BLE001
                _log(state, f"⚠️ could not read {state['ticket_key']}: {e}")
        return state

    # Node 1 — validate
    def validate_request(state: DeployState) -> DeployState:
        missing = [f for f in REQUIRED_FIELDS if not state.get(f)]
        env = (state.get("environment") or "").lower()
        if missing:
            state["status"] = "VALIDATION_FAILED"
            state["error"] = f"missing required fields: {missing}"
        elif env not in VALID_ENVIRONMENTS:
            state["status"] = "VALIDATION_FAILED"
            state["error"] = f"invalid environment '{env}' (allowed: {sorted(VALID_ENVIRONMENTS)})"
        else:
            state["environment"] = env
            _log(state, f"Validated {state['resource_type']}/{state['resource_name']} -> {env}.")
        return state

    # Node 2 — Azure audit (does the resource already exist?)
    def azure_audit(state: DeployState) -> DeployState:
        state["resource_exists"] = _azure_resource_exists(
            user_ws, state["resource_name"], state["resource_type"], state["environment"]
        )
        _log(state, f"Azure audit: resource {'FOUND' if state['resource_exists'] else 'NOT found'}.")
        return state

    # Node 3 — approval gate (staging/prod only). Approval is sourced from a human/Jira, never self-granted.
    def approval_gate(state: DeployState) -> DeployState:
        env = state["environment"]
        if env in GATED_ENVIRONMENTS:
            expected = f"CONFIRM-DEPLOY-{env}"
            if not state.get("authorized"):
                state["status"] = "PENDING_APPROVAL"
                state["error"] = (f"{state.get('actor')} is not authorized to deploy to {env} "
                                  f"(needs '{AUTHORIZED_DEPLOY_GROUP}').")
            elif state.get("approval_confirmation") != expected:
                state["status"] = "PENDING_APPROVAL"
                state["error"] = (f"human approval required for {env}: re-issue with "
                                  f"approval_confirmation='{expected}'. Do NOT self-approve.")
            else:
                _log(state, f"Approval verified for {env} by {state.get('actor')}.")
        else:
            _log(state, f"{env} is ungated; no approval required.")
        return state

    # Node 4 — trigger Terraform via GitHub Actions workflow_dispatch (async; runs on GH runners)
    def trigger_terraform(state: DeployState) -> DeployState:
        module = MODULE_FOR.get(state["resource_type"], state["resource_type"])
        try:
            gh.call_tool(GH_RUN_WORKFLOW, {
                "owner": IAC_REPO.split("/")[0], "repo": IAC_REPO.split("/")[1],
                "workflow_id": TERRAFORM_WORKFLOW, "ref": "main",
                "inputs": {"module_to_run": module, "environment": state["environment"],
                           "resource_name": state["resource_name"]},
            })
            state["tf_dispatched"] = True
            _log(state, f"Terraform dispatched (module={module}, env={state['environment']}).")
        except Exception as e:  # noqa: BLE001
            state["status"], state["error"] = "DEPLOY_FAILED", f"terraform dispatch failed: {e}"
        return state

    # Node 5 — render K8s manifest + open a GitOps PR via GitHub MCP
    def create_gitops_pr(state: DeployState) -> DeployState:
        if state.get("status") == "DEPLOY_FAILED":
            return state
        manifest = _render_manifest(state)
        path = f"clusters/{state['environment']}/{state['resource_name']}/deployment.yaml"
        branch = f"deploy/{state['resource_name']}-{state['environment']}"
        owner, repo = GITOPS_REPO.split("/")
        try:
            gh.call_tool(GH_CREATE_BRANCH, {"owner": owner, "repo": repo, "branch": branch, "from_branch": "main"})
            gh.call_tool(GH_PUT_FILE, {"owner": owner, "repo": repo, "branch": branch, "path": path,
                                       "content": manifest, "message": f"deploy {state['resource_name']} to {state['environment']}"})
            pr = gh.call_tool(GH_CREATE_PR, {"owner": owner, "repo": repo, "head": branch, "base": "main",
                                             "title": f"Deploy {state['resource_name']} -> {state['environment']}",
                                             "body": f"Automated by hon-deployment-agent for {state.get('ticket_key','(no ticket)')}"})
            state["pr_url"] = getattr(pr, "content", str(pr))
            state["status"] = "DEPLOYED"
            _log(state, f"GitOps PR opened: {path}")
        except Exception as e:  # noqa: BLE001
            state["status"], state["error"] = "DEPLOY_FAILED", f"gitops PR failed: {e}"
        return state

    # Node 6 — write status back to the Jira deployment ticket
    def update_jira(state: DeployState) -> DeployState:
        if not state.get("ticket_key"):
            return state
        target = {"DEPLOYED": "Deployed", "DEPLOY_FAILED": "Deploy Failed"}.get(state.get("status"))
        note = state.get("pr_url") or state.get("error") or state.get("status")
        try:
            if target:
                jira.call_tool(JIRA_TRANSITION, {"issueIdOrKey": state["ticket_key"], "status": target})
            jira.call_tool(JIRA_COMMENT, {"issueIdOrKey": state["ticket_key"],
                                          "body": f"hon-deployment-agent: {state.get('status')} — {note}"})
            state["jira_updated"] = True
        except Exception as e:  # noqa: BLE001
            _log(state, f"⚠️ Jira write-back failed: {e}")
        return state

    # Terminal — validation failure or pending approval
    def human_intervene(state: DeployState) -> DeployState:
        _log(state, f"HUMAN INTERVENTION [{state.get('status')}]: {state.get('error')}")
        # (optional) notify Slack via system.ai.slack MCP here.
        return state

    # ---- wiring -----------------------------------------------------------------
    g = StateGraph(DeployState)
    for name, fn in [("load_ticket", load_ticket), ("validate", validate_request),
                     ("azure_audit", azure_audit), ("approval_gate", approval_gate),
                     ("trigger_terraform", trigger_terraform), ("create_gitops_pr", create_gitops_pr),
                     ("update_jira", update_jira), ("human_intervene", human_intervene)]:
        g.add_node(name, fn)

    g.set_entry_point("load_ticket")
    g.add_edge("load_ticket", "validate")
    g.add_conditional_edges("validate",
        lambda s: "human_intervene" if s.get("status") == "VALIDATION_FAILED" else "azure_audit")
    g.add_edge("azure_audit", "approval_gate")
    g.add_conditional_edges("approval_gate",
        lambda s: "human_intervene" if s.get("status") == "PENDING_APPROVAL"
        else ("create_gitops_pr" if s.get("resource_exists") else "trigger_terraform"))
    g.add_edge("trigger_terraform", "create_gitops_pr")
    g.add_edge("create_gitops_pr", "update_jira")
    g.add_edge("update_jira", END)
    g.add_edge("human_intervene", END)
    return g.compile()


def _azure_resource_exists(ws: WorkspaceClient, name: str, rtype: str, env: str) -> bool:
    """Check Azure Resource Graph for an existing resource. Placeholder — wire to a real query
    (Azure SDK / a custom MCP / a UC function). Returning False means 'provision via Terraform'."""
    return False  # ⚠️ implement before production


# ----------------------------------------------------------------------------------
# The agent (governed ResponsesAgent shell)
# ----------------------------------------------------------------------------------
class DeploymentAgent(ResponsesAgent):
    def __init__(self) -> None:
        self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT, use_ai_gateway=True, temperature=0)  # ⚠️ verify kwarg

    def _prepare(self, request: ResponsesAgentRequest) -> tuple[DeployState, WorkspaceClient]:
        params: dict = dict(getattr(request, "custom_inputs", None) or {})
        # allow "...ticket HON-1234..." in free text to at least carry the ticket key
        if not params.get("ticket_key"):
            for item in request.input:
                text = item.get("content", "") if isinstance(item, dict) else ""
                for tok in str(text).replace(":", " ").split():
                    if "-" in tok and tok.split("-")[0].isalpha() and tok.split("-")[-1].isdigit():
                        params["ticket_key"] = tok
                        break
        user_ws = _obo_workspace_client()
        actor, groups = _current_user_and_groups(user_ws)
        params["actor"] = actor
        params["authorized"] = (AUTHORIZED_DEPLOY_GROUP in groups) or (actor in SERVICE_APPROVER_PRINCIPALS)
        params.setdefault("log", [])
        return params, user_ws  # type: ignore[return-value]

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        initial, user_ws = self._prepare(request)
        final = _build_graph(user_ws).invoke(initial)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=_summary(final), id="msg-0")],
            custom_outputs={k: final.get(k) for k in
                            ("status", "pr_url", "tf_dispatched", "resource_exists", "ticket_key", "error")},
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        initial, user_ws = self._prepare(request)
        graph = _build_graph(user_ws)
        final: Optional[DeployState] = None
        emitted = 0
        for state in graph.stream(initial, stream_mode="values"):  # full state after each node
            final = state
            log = state.get("log", [])
            while emitted < len(log):
                yield ResponsesAgentStreamEvent(
                    type="response.output_text.delta",
                    item=self.create_text_delta(delta=log[emitted] + "\n", item_id="msg-0"),
                )
                emitted += 1
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=_summary(final or {}), id="msg-0"),
        )


def _summary(state: DeployState) -> str:
    status = state.get("status", "UNKNOWN")
    lines = [f"**Deployment {status}** for {state.get('resource_name','?')} -> {state.get('environment','?')}"]
    if state.get("ticket_key"):
        lines.append(f"Ticket: {state['ticket_key']}")
    if state.get("pr_url"):
        lines.append(f"GitOps PR: {state['pr_url']}")
    if state.get("tf_dispatched"):
        lines.append("Terraform: dispatched via GitHub Actions.")
    if state.get("error"):
        lines.append(f"Note: {state['error']}")
    if state.get("log"):
        lines.append("\nSteps:\n- " + "\n- ".join(state["log"]))
    return "\n".join(lines)


AGENT = DeploymentAgent()
set_model(AGENT)  # MLflow "models from code" entry point
