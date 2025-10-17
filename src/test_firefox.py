
from selenium import webdriver
from selenium.webdriver.firefox.service import Service


import os
os.environ['DISPLAY'] = ':1'   # force use of display :1

service = Service('/usr/local/bin/geckodriver')
options = webdriver.FirefoxOptions()
# options.headless = False  # explicitly run with GUI

with webdriver.Firefox(service=service, options=options) as driver:
    driver.get('https://www.google.com')
    print(driver.title)
