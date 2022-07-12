From pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN apt-get update; exit 0
        #&& rm -rf /var/lib/apt/lists/*

RUN apt-get -y install git

RUN mkdir project \
        && cd project \
        && git clone https://github.com/ThibautChataing/binaps_explore.git \
        && python -m pip install pandas scipy

COPY ./Data/data/and_synthetic_scale_10_10000_10_0.001_0.05.dat ./project/.
COPY ./github_explore/github_cyber_commit_over_week_without_empty_line_2022-07-06T10:15:00.656093.dat ./project/.

# docker build . --file "Dockerfile" -t tchataing/binaps:test
CMD ["python", "./project/binaps_explore/Binaps_code/main.py", "-i", "./project/github_cyber_commit_over_week_without_empty_line_2022-07-06T10:15:00.656093.dat" ,"--save_model"]