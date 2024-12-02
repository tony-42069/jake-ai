from azureml.core import Workspace, Run
from azureml.core.authentication import AzureCliAuthentication
import time

# Initialize authentication
cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace(
    subscription_id="7997d72b-c6cb-4a15-bd13-8ab652725526",
    resource_group="dsadellari-rg",
    workspace_name="jake-ai-workspace",
    auth=cli_auth
)

# Get the run
run = Run.get(ws, 'jake-yi34b-training_1733160010_3e8ae34e')

while run.status not in ['Completed', 'Failed']:
    print(f"Run status: {run.status}")
    metrics = run.get_metrics()
    if metrics:
        print("Current metrics:", metrics)
    time.sleep(60)  # Check every minute

print(f"Run finished with status: {run.status}")
