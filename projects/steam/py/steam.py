import requests, json, os, warnings
from functools import cached_property
import pandas as pd

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
