"""
Log, register, and deploy the Deployment Agent (Path B — Model Serving).

Run this from a Databricks notebook or with the Databricks CLI configured.
Mirrors build-guide §4 (Requirement/worker-agent deployment). ⚠️ = verify on HON's workspace.

    %pip install -r requirements.txt
    dbutils.library.restartPython()
"""

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from databricks import agents

from deployment_agent import LLM_ENDPOINT, _github_mcp_url, _atlassian_mcp_url, _uc_functions_mcp_url

# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------
CATALOG, SCHEMA, NAME = "hon_platform", "agents", "deployment"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{NAME}"

w = WorkspaceClient()
host = w.config.host

# ----------------------------------------------------------------------------------
# 1. Declare downstream resources (System auth policy) so the platform provisions
#    least-privilege credentials, plus OBO scopes (User auth policy).
# ----------------------------------------------------------------------------------
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]

# Pull the concrete resource grants for each managed MCP server automatically.
for url in [_github_mcp_url(host), _atlassian_mcp_url(host), _uc_functions_mcp_url(host)]:
    try:
        resources += DatabricksMCPClient(server_url=url, workspace_client=w).get_databricks_resources()
    except Exception as e:  # noqa: BLE001 — a server may not be reachable at log time
        print(f"⚠️ could not resolve resources for {url}: {e}")

system_policy = SystemAuthPolicy(resources=resources)
user_policy = UserAuthPolicy(
    api_scopes=[
        "serving.serving-endpoints",  # call the LLM / other endpoints as the user
        "sql",                        # UC tables/functions via SQL Statement Execution
        # ⚠️ add the exact OAuth scope(s) the github/slack/atlassian MCP services require
    ]
)

# ----------------------------------------------------------------------------------
# 2. Log (models-from-code) with tracing + auth policy
# ----------------------------------------------------------------------------------
with mlflow.start_run():
    logged = mlflow.pyfunc.log_model(
        name="deployment_agent",              # ⚠️ older MLflow uses artifact_path=
        python_model="deployment_agent.py",   # entry point calls set_model(AGENT)
        pip_requirements=[
            "mlflow>=3.1.3",
            "databricks-agents>=1.1.0",
            "databricks-langchain",
            "databricks-mcp",
            "langgraph",
            "langchain-core",
            "databricks-sdk",
            "jinja2",
        ],
        auth_policy=AuthPolicy(
            system_auth_policy=system_policy,
            user_auth_policy=user_policy,
        ),
    )

# ----------------------------------------------------------------------------------
# 3. Register to Unity Catalog
# ----------------------------------------------------------------------------------
mlflow.set_registry_uri("databricks-uc")
uc = mlflow.register_model(model_uri=logged.model_uri, name=UC_MODEL_NAME)

# ----------------------------------------------------------------------------------
# 4. Deploy to Model Serving (keep warm — control-plane dependency)
# ----------------------------------------------------------------------------------
deployment = agents.deploy(UC_MODEL_NAME, uc.version, scale_to_zero_enabled=False)
print("Query endpoint:", deployment.query_endpoint)

# ----------------------------------------------------------------------------------
# NEXT: register 'hon-deployment-agent' as a worker in the supervisor's databricks.yml
#       (build-guide Step A4) so the supervisor routes deploy requests here via OBO,
#       and set MCP Service Policies to "require approval" for deploy/rollback tools.
# ----------------------------------------------------------------------------------
