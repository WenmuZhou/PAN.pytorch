FROM ubuntu:18.04
USER root

RUN apt update && apt upgrade -y && apt clean
RUN apt -y install curl
RUN apt update && \
    apt -y install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.6 python3.6-dev python3.6-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*


RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py

RUN python3.6 -m pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /panpytorch

COPY requirements.txt ./

RUN apt install g++
RUN apt install build-essential

RUN apt-get update && apt-get -y install libgl1-mesa-glx
RUN apt-get install -y gcc
RUN apt-get install -y python3-dev

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install gcc7
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install Polygon3
RUN python3.6 -m pip install --no-cache-dir -r requirements.txt

COPY ./ ./

EXPOSE 8081

CMD PYTHONPATH=/panpytorch/predict.py python3.6 predict.py