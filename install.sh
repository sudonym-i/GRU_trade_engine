
python3 -m venv .venv
source .venv/bin/activate
pip install -r engine/requirements.txt


sudo apt-get update && sudo apt upgrade
sudo apt-get install libgtest-dev
sudo apt-get install cmake
sudo apt install libfmt-dev 
sudo apt install libcurl
cd engine/social_media_sentiment/web_scraper/build
cmake ..
make

cd ../../../..