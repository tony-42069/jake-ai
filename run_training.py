from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.authentication import AzureCliAuthentication

# Initialize authentication
cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace(
    subscription_id="7997d72b-c6cb-4a15-bd13-8ab652725526",
    resource_group="dsadellari-rg",
    workspace_name="jake-ai-workspace",
    auth=cli_auth
)

print("Got workspace:", ws.name)

# Get compute target
compute_target = ComputeTarget(workspace=ws, name="jake-ai")
print("Got compute target:", compute_target.name)

# Create a new environment
env = Environment.from_conda_specification("jake-training-env", "environment.yml")
env.docker.base_image = None
env.docker.base_dockerfile = "FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04"

# Create script config
script_config = ScriptRunConfig(
    source_directory='.',
    script='src/train.py',
    arguments=['--config', 'config/training_config.yaml'],
    compute_target=compute_target,
    environment=env
)

# Create experiment and submit run
experiment = Experiment(workspace=ws, name='jake-yi34b-training')
run = experiment.submit(script_config)
print(f"Submitted run ID: {run.id}")
