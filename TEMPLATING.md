# DAG Templating

This project uses Jinja2 templating to dynamically set the Docker image used in the Airflow DAG.

## Files

- `src/ml_pipeline_dag.py.j2` - Jinja2 template for the DAG
- `src/ml_pipeline_dag.py` - Generated DAG file (auto-generated, do not edit directly)

## Template Variables

- `docker_image` - The Docker image URI to use for Kubernetes pods

## CI/CD Process

During the CI/CD pipeline:

1. The Docker image is built and tagged with the commit SHA
2. The image is pushed to GitHub Container Registry 
3. `jinja2-cli` is used to template the DAG with the full image URI
4. The templated DAG is synchronized to the S3 bucket for Airflow

## Local Development

To template the DAG locally for testing:

```bash
pip install jinja2-cli
jinja2 src/ml_pipeline_dag.py.j2 -D docker_image="your-image:tag" > src/ml_pipeline_dag.py
```

## Template Usage

The template replaces the hardcoded image name:

```python
# Before (hardcoded)
image="ml-pipeline-airflow:v2"

# After (templated)
image="{{ docker_image }}"
```

This allows the CI/CD pipeline to inject the exact image that was built and tested in the same workflow run.
