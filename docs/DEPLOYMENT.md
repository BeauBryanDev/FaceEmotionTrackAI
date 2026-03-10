# Deployment Guide — Emotitron Web App on AWS

This document describes how to deploy **Emotitron** to production using:

- **AWS ECS + Fargate** — containerized backend (FastAPI)
- **AWS RDS (PostgreSQL + pgvector)** — managed database
- **AWS S3** — frontend static hosting + ML model storage
- **AWS ECR** — Docker image registry
- **AWS SSM Parameter Store** — secrets management
- **GitHub Actions** — CI/CD pipeline

---

## Architecture Overview

```
GitHub Actions
    │
    ├── Build Docker image → push to ECR
    ├── Build React frontend → deploy to S3
    └── Update ECS service → Fargate pulls new image

Internet → (CloudFront) → S3 (React SPA)
                       → ECS Fargate (FastAPI backend)
                                  → RDS PostgreSQL (pgvector)
                                  → S3 (ML model weights)
```

---

## Prerequisites

- AWS CLI configured (`aws configure`)
- Docker installed locally
- An existing AWS VPC with subnets
- An existing RDS PostgreSQL instance with `pgvector` extension

---

## Step 1 — Provision RDS PostgreSQL

### 1.1 Connect to the database

```bash
psql -h "<AWS RDS endpoint>" -U "<admin user>" -p 5432 -d postgres
```

### 1.2 Create application user and grant privileges

```sql
CREATE USER emotitron_user WITH PASSWORD 'secret-password';
GRANT ALL PRIVILEGES ON DATABASE remote-db TO emotitron_user;
```

### 1.3 Enable pgvector extension

```bash
psql -h "<AWS RDS Endpoint>" \
     -U postgres \
     -d remote-db \
     -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

> **Status**: RDS `emotitron` instance — OK · pgvector 0.8.0 — OK · `beauadmin` user — OK

---

## Step 2 — Configure Network Security Groups

### 2.1 Authorize your local IP to reach RDS (port 5432)

```bash
aws ec2 authorize-security-group-ingress \
    --group-id sg-YOUR_RDS_SG_ID \
    --protocol tcp \
    --port 5432 \
    --cidr 192.168.X.X/32 \
    --description "LOCAL_OPERATOR_UPLINK"
```

### 2.2 Create a dedicated Fargate Security Group

```bash
aws ec2 create-security-group \
    --group-name ecs-fargate-sg \
    --description "EMOTRACK_FARGATE_COMPUTE_NODE" \
    --vpc-id vpc-YOUR_VPC_ID
```

### 2.3 Allow Fargate tasks to reach RDS

```bash
aws ec2 authorize-security-group-ingress \
    --group-id sg-YOUR_RDS_SG_ID \
    --protocol tcp \
    --port 5432 \
    --source-group sg-YOUR_NEW_FARGATE_SG_ID \
    --description "FARGATE_INTERNAL_UPLINK"
```

### 2.4 Verify the Fargate security group exists before creating the ECS service

```bash
aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=emotitron-fargate-sg" \
    --query 'SecurityGroups[*].[GroupId,GroupName]' \
    --region us-east-1
```

---

## Step 3 — Create ECR Repository

Push the backend Docker image to AWS Elastic Container Registry.

```bash
aws ecr create-repository \
    --repository-name emotitron-backend \
    --region us-east-1
```

> **Status**: ECR `emotitron-backend` — OK

---

## Step 4 — Create ECS Cluster (Fargate)

### 4.1 Create the ECS service-linked IAM role (one-time setup)

```bash
aws iam create-service-linked-role \
    --aws-service-name ecs.amazonaws.com
```

### 4.2 Create the ECS cluster with Fargate capacity

```bash
aws ecs create-cluster \
    --cluster-name emotitron-cluster-1001 \
    --capacity-providers FARGATE \
    --default-capacity-provider-strategy \
        capacityProvider=FARGATE,weight=1 \
    --region us-east-1
```

> **Status**: ECS `emotitron-cluster` — OK · ACTIVE

---

## Step 5 — Create S3 Buckets

### 5.1 Frontend static hosting bucket

```bash
aws s3api create-bucket \
    --bucket emotitron-frontend-1001 \
    --region us-east-1
```

This bucket serves:
- The compiled React SPA (HTML/JS/CSS)
- The 4 ML model weights (`.onnx` files)

---

## Step 6 — Create IAM Task Execution Role

ECS Fargate tasks need an IAM role to pull images from ECR, read SSM secrets, and write logs to CloudWatch.

### 6.1 Create the trust policy

```bash
cat > /tmp/ecs-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
```

### 6.2 Create the role

```bash
aws iam create-role \
    --role-name emotitron-task-execution-role \
    --assume-role-policy-document file:///tmp/ecs-trust-policy.json \
    --region us-east-1
```

### 6.3 Attach required policies

```bash
# Core ECS task execution permissions (ECR pull, CloudWatch logs)
aws iam attach-role-policy \
    --role-name emotitron-task-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Read secrets from SSM Parameter Store
aws iam attach-role-policy \
    --role-name emotitron-task-execution-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess
```

### 6.4 Verify the role ARN

```bash
aws iam get-role \
    --role-name emotitron-task-execution-role \
    --query 'Role.Arn'
```

---

## Step 7 — Store Secrets in AWS SSM Parameter Store

Application secrets are stored as `SecureString` parameters and injected into the Fargate task at runtime. Never store secrets in the Docker image or the task definition JSON.

### 7.1 Generate a secure JWT secret key

```bash
openssl rand -base64 64
```

### 7.2 Upload secrets to SSM

```bash
aws ssm put-parameter \
    --name "/emotitron/DB_PASSWORD" \
    --value "secret-db-password" \
    --type SecureString \
    --region us-east-1

aws ssm put-parameter \
    --name "/emotitron/SECRET_KEY" \
    --value "your-generated-secret-key" \
    --type SecureString \
    --region us-east-1

aws ssm put-parameter \
    --name "/emotitron/DATABASE_URL" \
    --value "postgresql://emotitron_user:secret-password@aws-rds-endpoint:5432/database-name" \
    --type SecureString \
    --region us-east-1
```

### 7.3 Verify secrets were saved

```bash
aws ssm get-parameters-by-path \
    --path "/emotitron" \
    --region us-east-1 \
    --query 'Parameters[].Name'
```

---

## Step 8 — Register ECS Task Definition

The task definition (`task-definition.json`) specifies the container image, CPU/memory, port mappings, environment variables pulled from SSM, and the execution role.

```bash
aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region us-east-1
```

> The `task-definition.json` file is checked into the repository root and references the execution role ARN and SSM parameter names defined above.

---

## Step 9 — Get Subnet IDs

The ECS service must be placed in a subnet that can reach the RDS instance (same VPC).

```bash
aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=YOUR_VPC_ID" \
    --query 'Subnets[*].[SubnetId,AvailabilityZone,CidrBlock]' \
    --output table \
    --region us-east-1
```

Note the `SubnetId` values — you will need them in Step 10.

---

## Step 10 — Create ECS Service

Deploy the backend as a long-running Fargate service.

```bash
aws ecs create-service \
    --cluster emotitron-cluster \
    --service-name emotitron-backend-service \
    --task-definition emotitron-backend-task \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-ID-XXXX],securityGroups=[emotitron-fargate-sg],assignPublicIp=ENABLED}" \
    --region us-east-1
```

- `--desired-count 1` — one running container (scale up as needed)
- `assignPublicIp=ENABLED` — required if the subnet is public and no NAT gateway is configured
- Replace `subnet-ID-XXXX` with the subnet ID from Step 9

---

## Step 11 — Upload ML Model Weights to S3

The 4 ONNX model files are not checked into Git. They must be uploaded to S3 so the Fargate task can download them at startup.

```bash
# Upload all 4 models from local backend/ml_weights/
aws s3 cp backend/ml_weights/ s3://emotitron-frontend-beauland/ml_weights/ \
    --recursive \
    --region us-east-1
```

### Verify all 4 models are present

```bash
aws s3 ls s3://emotitron-frontend-beauland/ml_weights/ --recursive
```

Expected files:
- `scrfd_2.5g_bnkps.onnx` — face detection (SCRFD)
- `minifasnetv2.onnx` — liveness detection
- `arcface_r50.onnx` — face embeddings (ArcFace)
- `enet_b0_8_best_vgaf.onnx` — emotion recognition (EmotiEffLib)

---

## Step 12 — Configure GitHub Actions CI/CD

### 12.1 Add GitHub repository secrets

Navigate to: `Settings > Secrets and variables > Actions > New repository secret`

| Secret Name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM user secret key |
| `ECR_REGISTRY` | `401257550058.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY` | `emotitron-backend` |
| `ECS_CLUSTER` | `emotitron-cluster` |
| `ECS_SERVICE` | `emotitron-backend-service` |
| `ECS_TASK_DEFINITION` | `emotitron-backend-task` |
| `S3_BUCKET` | `emotitron-frontend-beauland` |
| `CLOUDFRONT_DIST_ID` | Your CloudFront distribution ID (created with CloudFront setup) |
| `VITE_API_URL` | Public URL of your ECS service, e.g. `https://your-alb-url/api/v1` |

### 12.2 CI/CD Pipeline behavior

On every push to `master`, GitHub Actions will:

1. **Backend** — build the Docker image, push to ECR, force a new ECS deployment (Fargate pulls the new image)
2. **Frontend** — run `npm run build` with `VITE_API_URL` injected, sync the `dist/` folder to S3, invalidate the CloudFront cache

---

## AWS Resources Summary

| Resource | Name | Status |
|---|---|---|
| RDS PostgreSQL | `emotitron` | Active |
| pgvector extension | `0.8.0` | Installed |
| DB user | `beauadmin` | Created |
| ECR repository | `emotitron-backend` | Active |
| ECS cluster | `emotitron-cluster-1001` | ACTIVE |
| S3 bucket | `emotitron-frontend-beauland` | Created |
| IAM role | `emotitron-task-execution-role` | Created |
| Fargate SG | `ecs-fargate-sg` | Created |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Fargate task fails to start | Missing SSM permissions on execution role | Attach `AmazonSSMReadOnlyAccess` to the task execution role |
| Task exits immediately | `DATABASE_URL` SSM parameter incorrect | Verify the connection string and RDS security group rules |
| Backend can't reach RDS | Fargate SG not allowed inbound on RDS SG | Re-run Step 2.3 with the correct Fargate SG ID |
| ML models not loading | Models missing from S3 or wrong path | Re-run Step 11 and verify S3 paths match `task-definition.json` |
| `vector` extension missing | pgvector not enabled on RDS | Re-run Step 1.3 |
| Frontend 404 / blank page | `VITE_API_URL` incorrect or CloudFront cache stale | Check GitHub secret value; invalidate CloudFront distribution |


## Create a Private Bicket to store ML models

```bash
aws s3api create-bucket \
    --bucket emotitron-models-private \
    --region us-east-1 \
    --create-bucket-configuration LocationConstraint=us-east-1
```

## Apply S3 Bucket Policies

```bash
aws s3api put-bucket-policy \
    --bucket emotitron-models-private \
    --policy file://bucket-policy.json
```

## Confirm models are stored in S3

```bash
aws s3 ls s3://emotitron-models-private/ --recursive
``` 


## Endpoint 1 - SSM:
```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-dfgjgfyjgyj5 \
    --vpc-endpoint-type Interface \
    --service-name com.amazonaws.us-east-1.ssmmessages \
    --subnet-ids subnet-fghhjjj \
    --security-group-ids sg-ccccccc \
    --private-dns-enabled \
    --region us-east-1
```

## Endpoint 2 - SSM Messages:

```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-dfgjgfyjgyj5 \
    --vpc-endpoint-type Interface \
    --service-name com.amazonaws.us-east-1.ssmmessages \
    --subnet-ids subnet-fghhjjj \
    --security-group-ids sg-ccccccc \
    --private-dns-enabled \
    --region us-east-1
```

## Endpoint 3 - SSM Messages:

```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-dfgjgfyjgyj5 \
    --vpc-endpoint-type Interface \
    --service-name com.amazonaws.us-east-1.ssmmessages \
    --subnet-ids subnet-fghhjjj \
    --security-group-ids sg-ccccccc \
    --private-dns-enabled \
    --region us-east-1
```

## Endpoint 4 - SSM Messages:

```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-dfgjgfyjgyj5 \
    --vpc-endpoint-type Interface \
    --service-name com.amazonaws.us-east-1.ssmmessages \
    --subnet-ids subnet-fghhjjj \
    --security-group-ids sg-ccccccc \
    --private-dns-enabled \
    --region us-east-1
```
##  Check if all endpoints are avialable:

```bash
aws ec2 describe-vpc-endpoints \
    --filters "Name=vpc-id,Values=vpc-dffggfgjh!" \
    --query 'VpcEndpoints[*].{Service:ServiceName,State:State}' \
    --region us-east-1

```
## Check if all endpoints are avialable:

```bash
aws ec2 describe-vpc-endpoints \
    --filters "Name=vpc-id,Values=vpc-dffggfgjh!" \
    --query 'VpcEndpoints[*].{Service:ServiceName,State:State}' \
    --region us-east-1

```

## Check status of ECS service
```bash
aws ecs describe-services \
    --cluster emotitron-cluster \
    --services emotitron-backend-service \
    --query 'services[0].{Status:status,Running:runningCount,Pending:pendingCount,Desired:desiredCount}' \
    --region us-east-1

```
##  Check Tasks logs status

```bash
 aws ecs list-tasks \
    --cluster emotitron-cluster \
    --service-name emotitron-backend-service \
    --region us-east-1 \
    --query 'taskArns'
```
## Check Security Output to Internet 

```bash
aws ec2 describe-security-groups \
    --group-ids sg-sdfghjkl \
    --query 'SecurityGroups[0].{Inbound:IpPermissions,Outbound:IpPermissionsEgress}' \
    --region us-east-1

```


