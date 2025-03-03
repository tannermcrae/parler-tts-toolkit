#!/bin/bash

# Check if container type is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <container-type>"
    echo "Available container types: inference, train"
    exit 1
fi

CONTAINER_TYPE=$1

# Validate container type
if [ "$CONTAINER_TYPE" != "inference" ] && [ "$CONTAINER_TYPE" != "train" ]; then
    echo "Error: Invalid container type. Must be either 'inference' or 'train'"
    exit 1
fi

# Move to project root
cd "$(dirname "$0")/.."

# Configuration
AWS_REGION="us-west-2"  # Replace with your AWS region
ECR_REPO_NAME="parler-tts-${CONTAINER_TYPE}"  # Repository name based on container type
IMAGE_TAG="latest"  # Or you could use a version number or git commit hash

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

# Authenticate Docker with ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION || \
    aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION

# Build Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -t $ECR_REPO_URI:$IMAGE_TAG -f docker/Dockerfile.${CONTAINER_TYPE} .

# Tag image for ECR
# docker tag $ECR_REPO_NAME:$IMAGE_TAG $ECR_REPO_URI:$IMAGE_TAG

# Push to ECR
echo "Pushing to ECR..."
docker push $ECR_REPO_URI:$IMAGE_TAG

echo "Done! Image pushed to: $ECR_REPO_URI:$IMAGE_TAG"

echo "URI:$ECR_REPO_URI:$IMAGE_TAG"
