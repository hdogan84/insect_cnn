"""API crawler iNaturalist"""

"""
Downloads n images for a given species. The user only needs to provide the species id in the url 
and a target folder where the images should be downloaded.

"""


# Import libraries
import time
import requests
import io, os
from PIL import Image


def get_urls_from_api(taxa_id:int, page_num: int = 1, per_page: int = 5):
    """Find all image urls from a given observation page index.
    
    :param href_id: Observation page index. Leads to the page containing all images of a single person's contribution.
    :type href_id: int
    :param wd: Webdriver specific for your browser.
    :type wd: selenium.webdriver
    :param sleep: Number of seconds to wait until next iteration. Defaults to 5 seconds.
    :type sleep: int, optional
    :return: Set of tuples (urls, hrefID_listID)
    """
    
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxa_id,   
        "per_page": per_page, # Max per page seems to be 200
        "page": page_num,
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    image_urls = set()    


    for obs in data["results"]:
        #print(obs["species_guess"])
        #print(obs["quality_grade"])
        #print(obs["time_observed_at"])

        if len(obs["photos"]) == 0:
            continue

        square_url = obs["photos"][0]["url"]

        # check if it is a list or one single url
        assert type(square_url) == str

        large_url = square_url.replace('square', 'large') # store the image in original dimensions, not thumbnail dims
        image_urls.add(large_url)
    
    return image_urls

def persist_image(folder_path:str, url = str):
    """Save image from url to a specified folder.
    
    :param folder_path: Path to the folder where the images are saved.
    :type folder_path: str
    :param url_id: Address of the image to download and hrefID_listID
    :type url: tuple
    """

    img_id = url.split("/")[-2]
        
    
    try:
        # Get html code of the image
        headers = {'User-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0)'}
        image_content = requests.get(url, headers=headers).content
        
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")    
        
    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        
        spec = folder_path.split('\\')[-1]
        file_path = os.path.join(folder_path, spec + '_' + img_id + '.jpg')
                
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality = 95)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def search_and_download(search_ind:int , page_num: int=1, target_path = './', number_images = 10):
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

    # fetch image urls
    res = get_urls_from_api(search_ind , page_num, number_images)

        
    # Download images
    for elem in res:
        persist_image(target_path, elem)


        
if __name__ == '__main__':
    
    ind_spec = [
        (54327, 'Vespa_crabro'),
        (84804, 'Graphosoma_italicum'),
        (57516, 'Bombus_terrestris'),
        (52482, 'Episyrphus_balteatus'),
        (55719, 'Eristalis_tenax'),
        (126155, 'Vespula_germanica'),
        (84778, 'Leptinotarsa_decemlineata'),
        (47219, 'Apis_mellifera'),
        (52402, 'Cetonia_aurata'),
        (57619, 'Bombus_lapidarius'),
        (124550, 'Sarcophaga_carnaria'),
        (52488, 'Syrphus_ribesii'),
        (51699, 'Panorpa_communis'),   
        (52160, 'Scaeva_pyrastri'),
        (84640, 'Polistes_dominula'),
        (124145, 'Xylocopa_violacea')
    ]

    ind_spec = [
        (325997, 'Bombus_muscorum'),
    ]

    ind_spec = [
        (61968, 'Graphosoma_lineatum'),
    ]


    page_num = 2
    per_page = 200


    for ind, spec in ind_spec:
        
        print('\n******** ' + spec + ' ********\n')
        search_and_download(search_ind = ind,
                            #target_path = 'Z:\data\Bees\\' + spec, 
                            #target_path = 'C:\\Users\dgnhk\\Insect-CNN\data\image_data\\' + spec,
                            target_path = '/home/hakandogan/projects/Insect-CNN/data/image_data/' + spec,
                            page_num = page_num,
                            number_images = per_page)

    
    
    
    
    
    
    
    
    
    
    
     