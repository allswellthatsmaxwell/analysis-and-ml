import requests, json, os, warnings
from functools import cached_property
import pandas as pd
from typing import List, Union

APP_IDS_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"

class AppIDReader:
    json_path = "../data/app_ids.json"

    def _read_app_ids(self):
        if not os.path.exists(self.json_path):
            print(f"{self.json_path} not found... pulling from URL... ", end='')
            self._pull_app_ids()
            print("done.")
        with open(self.json_path) as f: 
            return json.loads(f.read())
        
    def _pull_app_ids(self):
        response = requests.get(APP_IDS_URL)
        app_id_json_str = response.text
        with open(self.json_path, 'w') as f:
            f.write(app_id_json_str)

    def _get_app_ids_dict(self):
        appids_to_names = {}
        app_id_json = self._read_app_ids()
        for entry in app_id_json['applist']['apps']:
            appid, name = entry['appid'], entry['name']
            if appid in appids_to_names and appids_to_names[appid] != name:
                warnings.warn(f"Previously found name '{appids_to_names[appid]}' for appid {appid}, but now found another one: '{name}'.")
            appids_to_names[appid] = name
        return appids_to_names

    @cached_property
    def app_names_dat(self):
        appids_to_names = self._get_app_ids_dict()
        rows = []
        for appid, name in appids_to_names.items():
            rows.append((appid, name))
        return pd.DataFrame(rows, columns=('appid', 'name'))


class ReviewCounter:
    def __init__(self, dat):
        self.dat = dat

    @cached_property
    def counts_by_game_dat(self):
        return self._count_by_game(['name'])

    @cached_property
    def positive_negative_ratios_dat(self):
        counts_dat = self._count_by_game(['name', 'voted_up'])
        return (
            counts_dat
            .merge(counts_dat, on='name', suffixes=('_pos', '_neg'))
            .query("voted_up_pos != voted_up_neg")
            .query("voted_up_pos")
            .assign(rr=lambda d: d['prop_pos'] / d['prop_neg'])
            .sort_values('rr', ascending=False))

    def _count_by_game(self, groupers: Union[str, List[str]]) -> pd.DataFrame:
        return (
            self.dat
            .groupby(groupers)
            .size()
            .reset_index(name='reviews')
            .assign(total_reviews=lambda d: d['reviews'].sum(),
                    prop=lambda d: d['reviews'] / d['total_reviews'])
            .sort_values('prop', ascending=False))