FROM ubuntu:20.04
ADD . /strawberry_pipeline/


EXPOSE 8080
EXPOSE 8000

# Install general dependences

RUN apt-get update
RUN apt-get install -y screen
RUN apt-get install -y curl
RUN apt-get install -y unzip
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
# RUN apt-get install -y cmake
# RUN apt-get install -y git

RUN pip3 install torch>=1.10
RUN pip3 install torchvision>=0.11.1

# Download third parties

RUN pip3 install gdown
WORKDIR /strawberry_pipeline/
RUN gdown "https://drive.google.com/uc?id=1FrMAR3ncJH3DxqS91e9k7Bw4m-zQ0rBb"
RUN unzip -o strawberry_models.zip
RUN rm strawberry_models.zip

RUN pip3 install -r requirements.txt

RUN pip3 install streamlit
RUN pip3 install pyyaml

CMD screen -dmS backand bash -c "uvicorn server:app --reload";streamlit run main.py --server.port=8080
