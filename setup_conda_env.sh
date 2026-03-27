#!/usr/bin/env bash

ENV_FILE="env.yaml"
ENV_NAME="mp2_env"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists, updating..."
    conda env update --name "$ENV_NAME" --file "$ENV_FILE" --prune
else
    echo "Creating new environment $ENV_NAME..."
    conda env create --name "$ENV_NAME" --file "$ENV_FILE"
fi