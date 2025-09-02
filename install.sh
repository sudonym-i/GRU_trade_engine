
echo "Beginning setup.."
echo ""
read -p "Enter your FMP API key: " fmp_api_key
read -p "Enter your new username: " username
read -p "Enter your new password: " password
echo ""
echo "Creating .env file.."
touch .env
echo "FMP_API_KEY=${fmp_api_key}" >> .env
echo "USERNAME=${username}" >> .env
echo "PASSWORD=${password}" >> .env
echo ""

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