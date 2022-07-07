FROM python:3.10-slim

RUN export DEBIAN_FRONTEND=noninteractive && \
    pip install --no-cache-dir \
                    pylint \
                    matplotlib \
                    ipykernel \
                    jupyter \
                    torch==1.6.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html
