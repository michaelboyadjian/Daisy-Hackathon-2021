import random
import numpy as np
from typing import List, Dict, Optional, Tuple
import copy

from site_location import SiteLocationPlayer, Store, SiteLocationMap, euclidian_distances, attractiveness_allocation

class MaSheen(SiteLocationPlayer):

    def place_stores(self, slmap: SiteLocationMap, 
                     store_locations: Dict[int, List[Store]],
                     current_funds: float):

        best_stores = self.getMaxPopulationDensity(slmap, store_locations)
        self.calculate_profit(best_stores)

        best_stores = sorted(best_stores, reverse = True, key = lambda x: x.profit)
        selected_stores = []
        num_selected = 0
        i = 0
        
        all_stores = store_locations[self.player_id]
        operating_cost = 0
        for store in all_stores:
            operating_cost += self.config["store_config"][store.store_type]["operating_cost"] 
        while(num_selected<2 and i<len(best_stores)):
            store = best_stores[i]
            operating_cost += self.config["store_config"][store.store_type]["operating_cost"] 
            capital_cost = self.config['store_config'][store.store_type]['capital_cost']
            total_cost = capital_cost + operating_cost
            operating_cost = 0
            if(total_cost>current_funds):
                i+=1
                continue
            if selected_stores and selected_stores[0].pos == store.pos:
                i+=1
                continue
            selected_stores.append(Store((best_stores[i].pos[0], best_stores[i].pos[1]), best_stores[i].store_type))

            current_funds = current_funds-total_cost
            i+=1
            num_selected+=1
            
        self.stores_to_place = selected_stores

    def getPopulationIndex(self, size, center):
        distances = euclidian_distances(size, center)
        indices = np.argwhere(distances <= 35)
        return indices

    def getStoreIndex(self, slmap):
        population = slmap.population_distribution.copy()
        Store_indices = []
        max_index = 0
        for i in range(20):
            coordinates = np.where(population == np.amax(population))
            max_index = list(zip(coordinates[0], coordinates[1]))[0]
            Store_indices.append(max_index)
        
            max_range = self.getPopulationIndex(slmap.size, max_index)
            for point in max_range:
                population[point[0], point[1]] = 0
        return Store_indices
        
    def getAttractivenessAllocationScore(self, slmap, store_conf, pos, store_type, store_locations):
        sample_store = Store(pos, store_type)
        temp_store_locations = copy.deepcopy(store_locations)
        temp_store_locations[self.player_id].append(sample_store)
        sample_alloc = attractiveness_allocation(slmap, temp_store_locations, store_conf)
        sample_score = (sample_alloc[self.player_id] * slmap.population_distribution).sum()
        return sample_score
        
    def getMaxPopulationDensity(self, slmap, store_locations):

        small_stores = []
        medium_stores = []
        large_stores = []
        indices = self.getStoreIndex(slmap)

        prev = []
        if store_locations[self.player_id]:
            for i in store_locations[self.player_id]:
                prev += [i.pos]
        
        for coordinate in indices:  
            if coordinate not in prev:
                store_conf = self.config['store_config']            
                small_store_score = self.getAttractivenessAllocationScore(slmap, store_conf, coordinate, "small", store_locations)
                new_small = MaSheenStore(pos=coordinate, store_type="small", score=small_store_score, population_density=None, population=None, profit=None)
                small_stores += [new_small]
                medium_store_score = self.getAttractivenessAllocationScore(slmap, store_conf,coordinate, "medium", store_locations)
                new_medium = MaSheenStore(pos=coordinate, store_type="medium", score=medium_store_score, population_density=None, population=None, profit=None)
                medium_stores += [new_medium]
                large_store_score = self.getAttractivenessAllocationScore(slmap, store_conf,coordinate, "large", store_locations)
                new_large = MaSheenStore(pos=coordinate, store_type="large", score=large_store_score, population_density=None, population=None, profit=None)
                large_stores += [new_large]

        best_small_stores = sorted(small_stores, key=lambda x: x.score, reverse=True)[0:3]
        best_medium_stores = sorted(medium_stores, key=lambda x: x.score, reverse=True)[0:3]
        best_large_stores = sorted(large_stores, key=lambda x: x.score, reverse=True)[0:3]
        rval = best_small_stores+best_medium_stores+best_large_stores

        return rval

    def calculate_profit(self, best_stores):
        for i in range(len(best_stores)):
            store_conf = self.config['store_config']
            if (best_stores[i].store_type == "small"):
                store_attributes = store_conf['small']
            if(best_stores[i].store_type == "medium"):
                store_attributes = store_conf['medium']
            if(best_stores[i].store_type == "large"):
                store_attributes = store_conf['large']
            revenue = best_stores[i].score * 0.5
            profit = revenue - store_attributes.get("capital_cost") - store_attributes.get("operating_cost")
            best_stores[i].profit = profit

class MaSheenStore:
    def __init__(self, pos, store_type, score, population_density, population, profit):
        self.pos = pos
        self.store_type = store_type
        self.score = score
        self.population_density = 0
        self.population = 0
        self.profit = 0