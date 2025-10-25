"""Webscraper iNaturalist"""

"""
Downloads n images for a given species. The user only needs to provide the species id in the url 
and a target folder where the images should be downloaded.

Code inspired by: 
    https://medium.com/swlh/web-scraping-stock-images-using-google-selenium-and-python-8b825ba649b9
    https://medium.com/geekculture/scraping-images-using-selenium-f35fab26b122
"""



# Import libraries
import time
import requests
import io, os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from PIL import Image
import pandas as pd


# Webdriver for Firefox downloaded with GeckoDriverManager. For other browsers, search for the specific webdriver service
# https://github.com/mozilla/geckodriver/releases





        

        
  
if __name__ == '__main__':
    
    # Test server: a browser window should open and close immediately
    # from selenium.webdriver.firefox.service import Service
    # service = Service(DRIVER_PATH)
    # service.start()
    # wd = webdriver.Remote(service.service_url)
    # wd.quit()
    
    # Wildbiene
    
    
    ind_spec = [
        #(54327, 'Vespa_crabro'),
        #(84804, 'Graphosoma_italicum'),
        #(57516, 'Bombus_terrestris'),
        #(52482, 'Episyrphus_balteatus'),
        #(55719, 'Eristalis_tenax'),
        #(126155, 'Vespula_germanica'),
        #(84778, 'Leptinotarsa_decemlineata'),
        #(47219, 'Apis_mellifera'),
        #(52402, 'Cetonia_aurata'),
        #(57619, 'Bombus_lapidarius'),
        (124550, 'Sarcophaga_carnaria'),
        #(52488, 'Syrphus_ribesii'),
        #(51699, 'Panorpa_communis'),   
        #(52160, 'Scaeva_pyrastri'),
        #(84640, 'Polistes_dominula'),
        #(124145, 'Xylocopa_violacea')
    ]

    df = pd.read_csv("../webapp_species/gbif_specie_taxon.csv")
    df_inat = pd.read_csv("/home/hakandogan/KInsektDaten_data/iNat/iNat_all_Insecta_classes_IS.csv")

    df_inat["name"] = df_inat["name"].str.replace(" ", "_")
    df["name"] = df["Latine_name"].str.replace(" ", "_")
    print(df_inat["name"][0:10])
    print(df["name"][0:10])

    df["in_iNat"] = df["name"].isin(df_inat["name"]).astype(int)
    
    df.to_csv("../webapp_species/gbif_specie_taxon_iNat.csv")
    
    total_exists = df["in_iNat"].sum()

    print(total_exists)
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     