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


        
if __name__ == '__main__':
    
    
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
        #(124145, 'Xylocopa_violacea')
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
        #(124550, 'Sarcophaga_carnaria'),
        #(52488, 'Syrphus_ribesii'),
        #(51699, 'Panorpa_communis'),   
        #(52160, 'Scaeva_pyrastri'),
        #(84640, 'Polistes_dominula'),
        #(124145, 'Xylocopa_violacea')
    ]

    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": 84640,   # Insects
        "per_page": 5, # Max per page seems to be 200
        "page": 1, # starts from 1, not 0
        "order_by": "created_at"
    }

    url = "https://api.inaturalist.org/v1/taxa"
    params = {
        "taxon_name": "Carabus_auratus",   # Insects
    }

    response = requests.get(url, params=params)
    data = response.json()

    #print(data["results"][0].keys())
    print(type(data))
    print(data.keys())
    print(data["total_results"])
    print(data["page"])
    print(len(data["results"]))
    print(data["results"][0].keys())
    print(data["results"][0]["iconic_taxon_id"])
    print(data["results"][0]["observations_count"])
    print(data["results"][0]["preferred_common_name"])
    #print(data["results"][0]["photos"])

   
    """for obs in data["results"]:
        #print(obs["id"], obs["species_guess"], obs["photos"])
        #print(obs["id"])
        print(len(obs["photos"]))
        print(obs["species_guess"])
        print(obs["quality_grade"])
        print(obs["time_observed_at"])
        print(obs["quality_metrics"])"""
    
    #print(data["results"][0]["photos"][0].keys())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     