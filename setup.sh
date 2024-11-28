# 리눅스 업데이트 및 git, tmux 설치
apt-get update && apt-get install wget
apt-get update; apt-get install build-essential ffmpeg libsm6 libxext6  -y
apt-get install tmux
apt-get install git
apt-get install tree

# crontab 설정
apt-get install cron
apt-get install vim
export EDITOR=vim
service cron start

# requirements.txt 설치
cd ./pytorch
pip install -r requirements.txt

echo "Setup completed"