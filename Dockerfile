                                                                                          From pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN apt-get update; exit 0
        #&& rm -rf /var/lib/apt/lists/*

RUN apt-get -y install git

# To run on OVH AI training the user 42420 should own the workspace
RUN mkdir -p /workspace && chown -R 42420:42420 /workspace
ENV HOME /workspace
WORKDIR /workspace

RUN git clone -b ovh_ai_training https://github.com/ThibautChataing/binaps_explore.git \
        && cd binaps_explore \
        && python -m pip install pandas scipy

COPY ./Data/data/and_synthetic_scale_10_10000_10_0.001_0.05.dat /workspace/.
COPY ./github_explore/github_cyber_commit_over_week_without_empty_line_2022-07-06T10:15:00.656093.dat /workspace/.


CMD ["python", \
    "/workspace/binaps_explore/Binaps_code/main.py", \
    "-i", \
    "/workspace/and_synthetic_scale_10_10000_10_0.001_0.05.dat", \
    "--save_model", \
    "--output_dir", \
    "/workspace/container_0"]


# To test as in ovh :
# docker build . --file "Dockerfile" -t tchataing/binaps:test
# docker run --rm -it --user=42420:42420 tchataing/binaps:test