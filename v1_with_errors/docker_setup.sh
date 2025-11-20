#!/bin/bash

echo "==== STOPPING all running containers ===="
docker stop $(docker ps -q) 2>/dev/null || true

echo "==== REMOVING all containers ===="
docker rm -f $(docker ps -aq) 2>/dev/null || true

echo "==== REMOVING all images ===="
docker rmi -f $(docker images -q) 2>/dev/null || true

# echo "==== REMOVING all volumes ===="
# docker volume rm $(docker volume ls -q) 2>/dev/null || true
echo "==== REMOVING all volumes except jenkins_home ===="
docker volume ls -q | grep -v '^jenkins_home$' | xargs -r docker volume rm

echo "==== REMOVING all NON-default networks ===="
docker network rm $(docker network ls | grep -v "bridge\|host\|none" | awk 'NR>1 {print $1}') 2>/dev/null || true

echo "==== ENSURING Docker Desktop is running ===="
open -a Docker
echo "Waiting for Docker to start..."
until docker info >/dev/null 2>&1; do
  sleep 2
  echo "Still waiting for Docker..."
done
echo "Docker is ready."

echo "==== CREATING mlops-net NETWORK ===="
docker network create mlops-net || true

echo "==== BUILDING your data-cleaner image ===="
docker build -t data-cleaner:local .

echo "==== RUNNING data-cleaner container ===="
docker run -d --name data-cleaner \
  --network mlops-net \
  -p 8000:8000 \
  data-cleaner:local

echo "==== VERIFYING data-cleaner is running ===="
docker ps --filter name=data-cleaner
curl -sS http://localhost:8000 || echo "App not responding yet."

echo "==== CREATING Jenkins volume ===="
docker volume create jenkins_home

echo "==== RUNNING Jenkins container ===="
docker run -d --name jenkins \
  --network mlops-net \
  -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -e DOCKER_HOST=tcp://host.docker.internal:2375 \
  jenkins/jenkins:lts

echo "==== INSTALLING docker CLI inside Jenkins ===="
docker exec -u root jenkins bash -lc "apt-get update && apt-get install -y docker.io"

echo "==== ADDING jenkins USER TO docker GROUP ===="
docker exec -u root jenkins bash -lc "usermod -aG docker jenkins || true"

echo "==== INSTALLING JENKINS PLUGINS (using modern jenkins-plugin-cli) ===="
docker exec -u root jenkins jenkins-plugin-cli --plugins "
  workflow-aggregator
  docker-workflow
  git
  github-branch-source
  credentials-binding
"

echo "==== RESTARTING Jenkins ===="
docker restart jenkins

echo "==== RETRIEVING Jenkins initial admin password ===="
sleep 8
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword

echo "==== NGROK: Ensure you have authtoken configured ===="
echo "Run if needed:  ngrok config add-authtoken <TOKEN>"
echo "==== STARTING NGROK ON PORT 8080 ===="
echo "ngrok http 8080"
