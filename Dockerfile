
########################< Docker File >########################
FROM vermasrijan/tf-dvc:latest

RUN apt update && apt upgrade
RUN apt install -y python3-pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY saved_models ./saved_models
COPY train.py config.py app.py ./

# For webapp

ENTRYPOINT ["python3", "app.py"]
EXPOSE 5000

#To run a docker container, Create an image using the Dockerfile.
docker build -t model .
docker build -t hrc-package .

#Run a container using the image
docker run hrc-package python3 train.py --dataset <path_to_private_data>
