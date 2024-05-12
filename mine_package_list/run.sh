#!/bin/bash

# Build the Docker image
docker build -t scraper .

# Ensure the docker_commands.txt is empty before starting
> docker_commands.txt

# Install GNU Parallel if it's not already installed
# sudo apt-get install -y parallel

mkdir -p ./scraper
# Read each package from the scrape_packages.txt and append Docker run commands to docker_commands.txt
while IFS= read -r package
do
  echo "docker run --rm -v \$(pwd)/scraper:/data scraper timeout 240 python parse.py --package $package --s3_bucket /data" >> docker_commands.txt
done < scrape_packages.txt

# Run the Docker commands in parallel
parallel -j 64 < docker_commands.txt

# clean up docker_commands.txt
rm docker_commands.txt