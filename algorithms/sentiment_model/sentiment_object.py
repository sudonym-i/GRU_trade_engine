# SentimentModel class for running sentiment analysis via C++ webscraper
import subprocess
import os

class SentimentModel:
	def __init__(self, scraper_path=None):
		# Default path to the compiled C++ binary
		if scraper_path is None:
			scraper_path = os.path.join(
				os.path.dirname(__file__),
				'web_scraper', 'build', 'webscrape.exe'
			)
		self.scraper_path = scraper_path

	def pull_from_web(self, *args):
		"""
		Execute the C++ webscraper binary with optional arguments.
		This method does not capture output; it only ensures the process runs successfully.
		Raises:
			FileNotFoundError: If the scraper binary does not exist.
			subprocess.CalledProcessError: If the process fails.
		"""
		if not os.path.isfile(self.scraper_path):
			raise FileNotFoundError(f"Scraper binary not found at {self.scraper_path}")
		cmd = [self.scraper_path] + list(args)
		subprocess.run(cmd, check=True)
