# Glaciation Forecast Service

This is a web service for obtaining short-term workload predictions on the Glaciation platform.

Upon committing and pushing, pre-commit triggers code checks and OpenAPI file generation.

Upon pushing the commit to GitHub, workflows are initiated, which:
- Check the code formatting;
- Execute server tests;
- Create a Docker image of the server, Helm chart, and deploy the application to a Kubernetes cluster.

## Requirements
Python 3.10+

## Installation
```bash
pip install pre-commit
pre-commit install
```

## Working on the server
Go to the `/server` folder to install dependencies and work on the server application.  
Documentation on setting up the virtual environment, installing dependencies, and working with the server can be found [here](./server/README.md).

## Usage
The service can be invoked by issuing an HTTP POST request with the [sample input data](./client/input.json) provided. For example:
```bash
curl -X POST -H "Content-Type: application/json" \
    --data-binary "@client/input.json" \
    http://forecast.integration/predict
```

## Release
The application version is specified in the VERSION file. The version should follow the format a.a.a, where 'a' is a number.  
To create a release, update the version in the VERSION file and add a tag in GIT.  
The release version for branches, pull requests, and tags will be generated based on the base version in the VERSION file.

## GitHub Actions
GitHub Actions triggers testing, builds, and application publishing for each release.  
https://docs.github.com/en/actions  

You can set up automatic testing in GitHub Actions for different versions of Python. To do this, you need to specify the set of versions in the `.github/workflows/server.yaml file`. For example:
```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
```

During the build and publish process, a Docker image is built, a Helm chart is created, an openapi.yaml is generated, and the web service is deployed to a Kubernetes cluster.

**Initial setup**  
1. Create the branch gh-pages and use it as a GitHub page https://pages.github.com/.  
2. Set up variables at `https://github.com/<workspace>/<project>/settings/variables/actions`:
- `DOCKER_IMAGE_NAME` - The name of the Docker image for uploading to the repository.

You can run your GitHub Actions locally using https://github.com/nektos/act. 
