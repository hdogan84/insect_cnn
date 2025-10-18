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


# Webdriver for Firefox downloaded with GeckoDriverManager. For other browsers, search for the specific webdriver service
# https://github.com/mozilla/geckodriver/releases
DRIVER_PATH = r'C:\Users\dgnhk\Downloads\geckodriver-v0.36.0-win64\geckodriver.exe'

DRIVER_PATH = r'C:\Users\dgnhk\.wdm\drivers\geckodriver\win64\v0.36.0\geckodriver.exe'

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep:int = 1):
    """Find and store the image urls.
    
    :param query: Species ID to complete the url.
    :type query: str
    :param max_links_to_fetch: Maximum number of urls to download.
    :type max_links_to_fetch: int
    :param wd: Webdriver specific for your browser.
    :type wd: selenium.webdriver
    :param sleep: Number of seconds to wait until next iteration. Defaults to 10 seconds.
    :type sleep: int, optional
    :return: Set of tuples (urls, hrefID_listID)
    """
    
    # Enable infinite scrolling
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep)  
         
    # Build the search query 
    # Only load fotos marked with Research Grade and CC-BY-NC copyright
    page_num = 1 # current page number
    if page_num == 1:
        search_url = f"https://www.inaturalist.org/observations?photo_license=CC-BY-NC&place_id=any&quality_grade=research&subview=table&taxon_id={query}"
    else:
        search_url = f"https://www.inaturalist.org/observations?page={page_num}&photo_license=CC-BY-NC&place_id=any&quality_grade=research&subview=table&taxon_id={query}"
    
    # Just for Sarcophaga carnaria
    search_url = f"https://www.inaturalist.org/observations?taxon_name={query}&quality_grade=research&subview=table"


    wd.get(search_url)
    time.sleep(sleep)  
    
    image_urls = set() # will contain tuples of urls along with the hrefID and listID within contribution: (url, hrefID_listID)
    # Note: by storing the href and list IDs, each downloaded image can be retraced exactly on the site
    image_count = 0 
    results_start = 0
    reached_max = False
    
    # Get total number of pages 
    try:
        page_links = WebDriverWait(wd, 30).until(EC.presence_of_all_elements_located((By.XPATH, "//li[@class='pagination-page ng-scope']/a")))                
        num_pages = int( page_links[-1].get_attribute('text') ) 
    except:
        num_pages = 1 # only one page, no links to new pages

    # View all results on page
    while image_count < max_links_to_fetch:
        
        scroll_to_end(wd)
        thumb = wd.find_elements(By.CSS_SELECTOR, 'a.img') # after each scrolling, the list increases
        num_results = len(thumb)
        
        # If the new thumbnail list is as long as before, you have reached the end of the page
        # Load next page (in case there is one)
        if num_results == results_start:
            
            if page_num < num_pages:
                page_num += 1
                search_url = f"https://www.inaturalist.org/observations?page={page_num}&quality_grade=research&subview=table&taxon_name={query}" 
                print('\nLoading next page...\n')
                wd.get(search_url)
                
                results_start = 0 # the new thumbnail list on the next page will be scanned from 0 again
                
                # The list of page links at the bottom can only show 10 pages at once.
                # If there are more than 10 pages, the links to the new ones are only shown as you
                # progress through the links. Therefore, whenever you turn a page, check
                # whether there are actually more pages than initially visible.
                try:
                    page_links = WebDriverWait(wd, 30).until(EC.presence_of_all_elements_located((By.XPATH, "//li[@class='pagination-page ng-scope']/a")))                
                    num_pages = int( page_links[-1].get_attribute('text') )              
                    continue
                
                except Exception as e:
                    print(f"ERROR - Could not find page links on page {page_num}. Returning urls stored until now - {e}")  
                    break
            
            else:
                print('No more images left!')
                break
                     
        # Iterate over (new) images in current thumbnail list
        for img in thumb[results_start : num_results]:
            
            href_att = img.get_attribute('href')
            href_id = href_att[href_att.rfind('/') + 1 : ] # drop string 'observations'

            taxon_id = get_taxon_id_from_href(href_id, 2)
            print(taxon_id)

            break
        break
            
        
    print(f"Taxon_id: {taxon_id}.  Done!\n")
    return image_urls, taxon_id

def get_taxon_id_from_href(href_id:int, sleep:int = 2):
    """Find all image urls from a given observation page index.
    
    :param href_id: Observation page index. Leads to the page containing all images of a single person's contribution.
    :type href_id: int
    :param wd: Webdriver specific for your browser.
    :type wd: selenium.webdriver
    :param sleep: Number of seconds to wait until next iteration. Defaults to 5 seconds.
    :type sleep: int, optional
    :return: Set of tuples (urls, hrefID_listID)
    """
    
    api_url = f"https://api.inaturalist.org/v1/observations/{href_id}"
    
    response = requests.get(api_url)
    data = response.json()
    
    time.sleep(sleep) 
    
    taxon_id = data["results"][0]["taxon"]["id"]
    
    return taxon_id


        
def search_and_download(search_term:str, target_path = './', number_images = 10):
    """Launch query, store urls and download images.
    
    :param search_term: Image ID used as query in the url.
    :type search_term: str
    :param target_path: Path to the folder where the images should be downloaded. Defaults to current folder.
    :type target_path: str, optional
    :param number_images: Number of images to be downloaded. Defaults to 10.
    :type number_images: int, optional
    """
    
    # Create downloading path, if not already existant
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Store image urls
    # Note: On the first run, install driver with GeckoDriverManager().install() instead of DRIVER_PATH
    with webdriver.Firefox(service = Service(DRIVER_PATH)) as wd:
        res, taxon_id = fetch_image_urls(search_term, number_images, wd = wd)

        
  
if __name__ == '__main__':
    
    # Test server: a browser window should open and close immediately
    # from selenium.webdriver.firefox.service import Service
    # service = Service(DRIVER_PATH)
    # service.start()
    # wd = webdriver.Remote(service.service_url)
    # wd.quit()
    
    # Wildbiene
    ind_spec = [
        #(60579, 'Andrena_fulva'),
        #(62453, 'Anthidium_manicatum'),
        #(453068, 'Bombus_cryptarum'),
        #(121989, 'Bombus_hortorum'),
        #(61803, 'Bombus_hypnorum'),
        #(57619, 'Bombus_lapidarius'),
        #(61856, 'Bombus_lucorum'),
        #(424468, 'Bombus_magnus'),
        #(55637, 'Bombus_pascuorum'),
        #(124910, 'Bombus_pratorum'),
        #(123657, 'Bombus_sylvarum'),
        #(746682, 'Dasypoda_hirtipes'),
        #(415589, 'Halictus_scabiosae'),
        #(207574, 'Osmia_bicolor'),
        #(876599, 'Osmia_bicornis'),
        #(126630, 'Osmia_cornuta'),
        #(154661, 'Sphecodes_albilabris'),
        (124145, 'Xylocopa_violacea')
    ]
    
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

    for ind, spec in ind_spec:
        
        print('\n******** ' + spec + ' ********\n')
        search_and_download(search_term = str(spec),
                            target_path = 'C:\\Users\\dgnhk\\insect_cnn\\data\\image_data\\' + spec,
                            number_images = 2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     