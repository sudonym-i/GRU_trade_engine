
# May need to automate model downloading as well


echo "Setting up Python virtual environment.."
sudo apt install python3-full
python3 -m venv .venv
source .venv/bin/activate
echo "Installing Python dependencies.."
pip install -r engine/requirements.txt
echo ""


echo "Setting up C++ environment.."
sudo apt-get update && sudo apt upgrade
sudo apt-get install libgtest-dev
sudo apt-get install cmake
sudo apt install libfmt-dev 
sudo apt install curl
sudo apt install libcurl4-gnutls-dev
echo "Compiling c++"
cd engine/sentiment_model/web_scraper/build
cmake ..
make
chmod +x webscrape.exe && echo "C++ build successful"
echo ""

cd ../../../..

echo "Setup and installation complete."
echo ""
echo ""