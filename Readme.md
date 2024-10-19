# Observability Platform logging benchmark

This repository contains code to run automated benchmarks for log aggregation with common observability platforms, like Elasticsearch and Loki.

## Running the benchmark

First copy the `inventory/inventory.sample.yaml` to `inventory/inventory.yaml` and fill in the necessary information.

Then run the following commands:

```shell
# Install dependencies
ansible-galaxy install -r collections/requirements.yaml
# Run the benchmark 
ansible-playbook main.yaml
```
When running the playbook on a mac, you might need to run the playbook with the environment variable `OBJC_DISABLE_INITIALIZE_FORK_SAFETY` set to `YES`:
```shell
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES ansible-playbook main.yaml
```