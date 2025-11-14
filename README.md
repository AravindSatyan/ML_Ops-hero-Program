instructions to run it in Mac

# start Docker Desktop (graphical app)
open -a Docker

# wait for Docker to initialize, then verify Docker daemon is reachable
# this may take a few seconds; run repeatedly until 'docker info' succeeds
docker info
# or quick check
docker version

docker network create mlops-net

# tag is local; pipeline builds tags differently when Jenkins runs
docker build -t data-cleaner:local .

docker run -d --name data-cleaner \
  --network mlops-net \
  -p 8000:8000 \
  data-cleaner:local

docker ps --filter name=data-cleaner
curl -sS http://localhost:8000/  # or the app's health endpoint if present

# create jenkins data volume
docker volume create jenkins_home

# run Jenkins (detached) on the mlops-net
docker run -d --name jenkins \
  --network mlops-net \
  -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts

  # install docker CLI inside the container (run as root)
docker exec -u root jenkins bash -lc "apt-get update && apt-get install -y docker.io"

# add the jenkins user to docker group (allow running docker without root)
docker exec -u root jenkins bash -lc "usermod -aG docker jenkins || true"

# install common Jenkins plugins needed for pipeline + multibranch + GitHub
docker exec -u root jenkins bash -lc "/usr/local/bin/install-plugins.sh workflow-aggregator docker-workflow git github-branch-source credentials-binding"
# restart Jenkins so plugins and docker-group change apply
docker restart jenkins

## after running the command below, copy the key given in the output
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword

# install ngrok (if needed)
brew install --cask ngrok

# Authenticate (replace with your ngrok authtoken)
# replace <NGROK_AUTH_TOKEN> with your token from ngrok.com

ngrok config add-authtoken <NGROK_AUTH_TOKEN> 

# open HTTP tunnel to Jenkins (8080)
ngrok http 8080
# ngrok will print a forwarding URL like https://abcd-1234.ngrok.io
# test commit
