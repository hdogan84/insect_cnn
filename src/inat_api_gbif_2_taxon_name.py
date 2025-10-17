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
        "order_by": "created_at"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    #print(data)

    image_urls = set()    

    if "results" not in data.keys():
        return image_urls, None

    if len(data["results"])==0 or "results" not in data.keys():
        return image_urls, None

    specie_name = data["results"][0]["taxon"]["name"].replace(" ","_")
    print(specie_name)

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
    
    return image_urls, specie_name




def search_and_download(search_ind:int , page_num: int=1, target_path = './', number_images = 10):
    """Launch query, store urls and download images.
    
    :param search_term: Image ID used as query in the url.
    :type search_term: str
    :param target_path: Path to the folder where the images should be downloaded. Defaults to current folder.
    :type target_path: str, optional
    :param number_images: Number of images to be downloaded. Defaults to 10.
    :type number_images: int, optional
    """

    # fetch image urls
    res, specie_name = get_urls_from_api(search_ind , page_num, number_images)

    return specie_name


        
if __name__ == '__main__':
    

    ind_spec = [1036779,1043097,1043717,1047536,1048809,1063891,1071240,1071266,1077829,1310622,1311477,1311527,1311649,1311651,1311671,1311676,1311797,1311798,1311805,1311825,1311831,1311838,1312367,1312773,1315132,1318848,1318874,1329873,1334278,1334493,1334499,1334821,1335321,1335556,1335648,1336116,1336802,1337092,1338442,1338640,1339951,1340286,1340301,1340305,1340342,1340344,1340394,1340405,1340418,1340434,1340503,1340527,1340542,1341976,1342108,1346919,1348620,1349795,1357156,1428217,1495562,1497483,1500013,1502577,1524843,1535521,1535529,1536289,1536449,1536559,1536796,1537212,1537245,1537266,1537412,1537717,1537719,1541217,1541740,1541763,1541799,1541832,1590998,1670212,1873079,2105360,4388646,4452252,4459737,4470628,4485729,4485776,4486826,4493809,4494295,4516754,4518735,4989904,4990191,4990995,4991026,4994132,4994160,5035785,5035857,5037317,5039039,5039096,5039166,5039314,5040875,5063973,5071172,5156102,5742495,5766554,5766556,5871389,5978844,7412043,8703749,9198953,9295353,
            9711771,11195063,11219325,
            ]

    page_num = 1
    per_page = 1

    print(len(ind_spec))

    for i in range(len(ind_spec)):

        ind = ind_spec[i]

        print('\n******** ' + str(ind) + ' ********\n')
        specie_name = search_and_download(search_ind = ind,
                            target_path = '/home/hakandogan/projects/Insect-CNN/data/image_data/',
                            page_num = page_num,
                            number_images = per_page)

        print(specie_name)

