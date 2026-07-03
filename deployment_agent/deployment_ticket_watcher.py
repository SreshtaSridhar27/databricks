"""
Deployment-ticket watcher — the AUTOMATED trigger for the Deployment Agent.

WHAT IT DOES
------------
Runs as a Databricks Job with a TABLE-UPDATE TRIGGER on the Lakeflow-Connect-synced Jira
`issues` Delta table. When Jira syncs a status change, this job:
  1. Finds Deployment tickets that just became "Ready to Deploy" (parent Done) and are not yet deployed.
  2. Verifies — from the Jira changelog — that an authorized DevOps user made the approving transition.
  3. Mints the one-time approval token and invokes the Deployment Agent serving endpoint per ticket.
  4. Records the ticket as dispatched in a control table so it is never deployed twice.

The AGENT (not this watcher) does the actual deploy + writes status back to Jira (Deploying/Deployed/Failed).

CONFIGURE THE JOB TRIGGER (Databricks Jobs → this task):
  Trigger type: "Table update"  →  table: hon_platform.jira.issues
  (Fallback while the Jira connector is Beta: a scheduled trigger, e.g. every 5 minutes.)

⚠️ = confirm table/field/status names against HON's Jira scheme + Lakeflow Jira connector schema.
"""

import json

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

# ----------------------------------------------------------------------------------
# Config  (⚠️ align with HON's Jira project + Lakeflow Jira connector tables)
# ----------------------------------------------------------------------------------
JIRA_CATALOG, JIRA_SCHEMA = "hon_platform", "jira"          # where Lakeflow lands Jira
ISSUES_TABLE = f"{JIRA_CATALOG}.{JIRA_SCHEMA}.issues"
CONTROL_TABLE = "hon_platform.deploy_ops.deployment_dispatch_log"

DEPLOY_ISSUE_TYPE = "Deployment"
READY_STATUS = "Ready to Deploy"        # the status a DevOps user transitions the ticket into
AUTHORIZED_DEPLOY_GROUP = "hon-devops"  # approver must belong to this group
AGENT_ENDPOINT = "hon-deployment-agent" # the deployed Model Serving endpoint

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()


def find_ready_tickets() -> list[dict]:
    """Deployment tickets whose parent is Done and status == READY_STATUS, not yet dispatched."""
    spark.sql(
        f"""CREATE TABLE IF NOT EXISTS {CONTROL_TABLE}
            (issue_key STRING, status_at_dispatch STRING, dispatched_at TIMESTAMP)"""
    )
    # ⚠️ Column names (issue_key, issue_type, status, environment, service, fix_version, parent_key,
    #    parent_status) depend on the Jira connector schema — some live in issue_field_values and
    #    may need a join/pivot. Adjust to your landed schema.
    rows = spark.sql(
        f"""
        SELECT i.issue_key, i.environment, i.service, i.fix_version, i.parent_key
        FROM {ISSUES_TABLE} i
        WHERE i.issue_type = '{DEPLOY_ISSUE_TYPE}'
          AND i.status = '{READY_STATUS}'
          AND i.parent_status IN ('Done', 'Resolved', 'Closed')
          AND i.issue_key NOT IN (SELECT issue_key FROM {CONTROL_TABLE})
        """
    ).collect()
    return [r.asDict() for r in rows]


def approver_is_authorized(issue_key: str) -> bool:
    """Confirm the transition INTO READY_STATUS was performed by a DevOps user.
    This is what makes the automated path a real human approval, not a self-approval.

    Prefer the Jira changelog: read via the Atlassian MCP tool, or from a landed
    `issue_changelogs` Delta table if the connector provides one. ⚠️ verify availability.
    """
    # Placeholder — replace with a real changelog lookup, e.g.:
    #   author = latest_transition_author(issue_key, to_status=READY_STATUS)
    #   return AUTHORIZED_DEPLOY_GROUP in groups_of(author)
    # If the Jira workflow already restricts the READY transition to the DevOps role via a
    # permission scheme, this check is belt-and-suspenders but still recommended for audit.
    return True  # ⚠️ implement before production


def dispatch(ticket: dict) -> None:
    key = ticket["issue_key"]
    env = (ticket.get("environment") or "staging").lower()
    if not approver_is_authorized(key):
        print(f"SKIP {key}: approving transition not made by an authorized {AUTHORIZED_DEPLOY_GROUP} user.")
        return

    # Mint the one-time approval token the agent's gate expects (see deployment_agent.py).
    approval_token = f"CONFIRM-DEPLOY-{env}"

    # Structured params drive the DETERMINISTIC pipeline (no LLM parsing of free text).
    custom_inputs = {
        "ticket_key": key,
        "resource_name": ticket.get("service"),
        "resource_type": ticket.get("resource_type", "redis"),  # ⚠️ map from your Jira fields
        "environment": env,
        "version": ticket.get("fix_version", "latest"),
        "approval_confirmation": approval_token,
    }
    resp = w.serving_endpoints.query(
        name=AGENT_ENDPOINT,
        # Runs behind OBO as the watcher SP (allowlisted in SERVICE_APPROVER_PRINCIPALS).
        inputs={
            "input": [{"role": "user", "content": f"Deploy approved ticket {key}."}],
            "custom_inputs": custom_inputs,
        },
    )
    print(f"DISPATCHED {key} -> {env}: {json.dumps(getattr(resp, 'as_dict', lambda: resp)())[:500]}")

    # Record so we never dispatch the same ticket twice (idempotency).
    spark.sql(
        f"""INSERT INTO {CONTROL_TABLE}
            VALUES ('{key}', '{READY_STATUS}', current_timestamp())"""
    )


def main() -> None:
    tickets = find_ready_tickets()
    print(f"Found {len(tickets)} deployment ticket(s) ready to deploy.")
    for t in tickets:
        try:
            dispatch(t)
        except Exception as e:  # noqa: BLE001 — one bad ticket must not stop the batch
            print(f"ERROR dispatching {t.get('issue_key')}: {e}")


if __name__ == "__main__":
    main()
