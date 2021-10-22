FROM python:3.8

WORKDIR /usr/src/app

COPY ./configs ./configs
COPY ./data ./data
COPY ./img ./img
COPY ./3rd-party-licenses.txt ./
COPY ./download_eval_scripts.sh ./
COPY ./environment.yml ./
COPY ./LICENSE ./
COPY ./README.md ./

RUN pip install wheel setuptools cython

RUN pip install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

COPY ./requirements_service.txt ./
RUN pip install -r requirements_service.txt

RUN pip install transformers==3.1.0
RUN pip install mlflow stanza pyconll

COPY ./src ./src

# disable torch gpu
ENV CUDA_VISIBLE_DEVICES=

EXPOSE 8000

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# only use 1 worker, multiple have not been tested with model
CMD ["uvicorn", "parse_service:app", "--host", "0.0.0.0", "--port" ,"8000", "--workers", "1", "--app-dir", "src/"]
