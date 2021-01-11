import random
import numpy as np
from typing import List, Dict, Optional, Tuple
import copy

from site_location import SiteLocationPlayer, Store, SiteLocationMap, euclidian_distances, attractiveness_allocation

class RandomPlayer(SiteLocationPlayer):
    """
    Player attempts to place the maximum stores, with each store type and
    position chosen randomly.
    """
    def place_stores(self, slmap: SiteLocationMap, 
                     store_locations: Dict[int, List[Store]],
                     current_funds: float):
        stores = []
        for _ in range(self.config["max_stores_per_round"]):
            store_types = list(self.config["store_config"].keys())
            store = Store((random.randrange(0, slmap.size[0]),
                           random.randrange(0, slmap.size[1])),
                          random.choice(store_types))
            stores.append(store)
        self.stores_to_place = stores

class MaxDensityPlayer(SiteLocationPlayer):
    """ 
    Player always selects the highest density location at least 50 units
    away from the nearest store. 

    Store type will always be the largest one it can afford.
    """
    def place_stores(self, slmap: SiteLocationMap, 
                     store_locations: Dict[int, List[Store]],
                     current_funds: float):
        store_conf = self.config['store_config']
        # Configurable minimum distance away to place store
        min_dist = 50
        # Check if it can buy any store at all
        if current_funds < store_conf['small']['capital_cost']:
            self.stores_to_place = []
            return
        # Choose largest store type possible
        if current_funds >= store_conf['large']['capital_cost']:
            store_type = 'large'
        elif current_funds >= store_conf['medium']['capital_cost']:
            store_type = 'medium'
        else:
            store_type = 'small'
        # Find highest population location
        all_stores_pos = []
        for player, player_stores in store_locations.items():
            for player_store in player_stores:
                all_stores_pos.append(player_store.pos)
        
        sorted_indices = tuple(map(tuple, np.dstack(np.unravel_index(np.argsort(slmap.population_distribution.ravel()), slmap.size))[0][::-1]))
        for max_pos in sorted_indices:
            too_close = False
            for pos in all_stores_pos:
                dist = np.sqrt(np.square(max_pos[0]-pos[0]) + np.square(max_pos[1]-pos[1]))
                if dist < min_dist:
                    too_close = True
            if not too_close:
                self.stores_to_place = [Store(max_pos, store_type)]

class CopycatPlayer(SiteLocationPlayer):
    """ 
    Player places an identical store at the location of a random opponent's store.
    """
    def place_stores(self, slmap: SiteLocationMap, 
                     store_locations: Dict[int, List[Store]],
                     current_funds: float):

        self_stores_pos = []
        for store in store_locations[self.player_id]:
            self_stores_pos.append(store.pos)

        opp_store_locations = {k:v for (k,v) in store_locations.items() if k != self.player_id}
        opp_all_stores = []
        for player, player_stores in opp_store_locations.items():
            for player_store in player_stores:
                if player_store.pos not in self_stores_pos:
                    opp_all_stores.append(player_store)
        if not opp_all_stores:
            self.stores_to_place =  []
            return
        else:
            self.stores_to_place = [random.choice(opp_all_stores)]
            return
        
class AllocSamplePlayer(SiteLocationPlayer):
    """
    Agent samples locations and selects the highest allocating one using
    the allocation function. 
    """
    def place_stores(self, slmap: SiteLocationMap, 
                     store_locations: Dict[int, List[Store]],
                     current_funds: float):
        store_conf = self.config['store_config']
        num_rand = 100

        sample_pos = []
        for i in range(num_rand):
            x = random.randint(0, slmap.size[0])
            y = random.randint(0, slmap.size[1])
            sample_pos.append((x,y))
        # Choose largest store type possible:
        if current_funds >= store_conf['large']['capital_cost']:
            store_type = 'large'
        elif current_funds >= store_conf['medium']['capital_cost']:
            store_type = 'medium'
        else:
            store_type = 'small'

        best_score = 0
        best_pos = []
        for pos in sample_pos:
            sample_store = Store(pos, store_type)
            temp_store_locations = copy.deepcopy(store_locations)
            temp_store_locations[self.player_id].append(sample_store)
            sample_alloc = attractiveness_allocation(slmap, temp_store_locations, store_conf)
            sample_score = (sample_alloc[self.player_id] * slmap.population_distribution).sum()
            if sample_score > best_score:
                best_score = sample_score
                best_pos = [pos]
            elif sample_score == best_score:
                best_pos.append(pos)

        # max_alloc_positons = np.argwhere(alloc[self.player_id] == np.amax(alloc[self.player_id]))
        # pos = random.choice(max_alloc_positons)
        self.stores_to_place = [Store(random.choice(best_pos), store_type)]
        return 

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
        num_stores = len(all_stores)
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
        #allocplayer = self.alloc_player(slmap, store_locations, store_conf)
        sample_store = Store(pos, store_type)
        temp_store_locations = copy.deepcopy(store_locations)
        temp_store_locations[self.player_id].append(sample_store)
        #temp_store_locations[self.player_id].append(allocplayer[0])
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

    def alloc_player(self, slmap, store_locations, store_conf):
        num_rand = 10
        sample_pos = []
        for i in range(num_rand):
            x = random.randint(0, slmap.size[0])
            y = random.randint(0, slmap.size[1])
            sample_pos.append((x,y))
        st = random.randint(0,1)
        if st == 1: 
            stype = "large"
        else:
            stype = "small"
        best_pos = []
        for pos in sample_pos:
            sample_store = Store(pos, stype)
            temp_store_locations = copy.deepcopy(store_locations)
            temp_store_locations[self.player_id].append(sample_store)
            sample_alloc = attractiveness_allocation(slmap, temp_store_locations, store_conf)
            sample_score = (sample_alloc[self.player_id] * slmap.population_distribution).sum()
            best_pos += [[pos, sample_score]]
        best_pos = sorted(best_pos, reverse = True, key = lambda x: x[1])
        best_pos = random.sample(best_pos[0:5], 2)
        output_stores = []
        for i in best_pos:
            output_stores += [Store(i[0], stype)]
        return output_stores

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