import requests, json, os, warnings
from functools import cached_property
import pandas as pd, numpy as np
from typing import List, Union
from scipy import sparse
from scipy.sparse import csr_matrix
import plotnine as pn


APP_IDS_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
DATA_DIR = '../data'


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
            if name is None or pd.isnull(name) or (type(name) != str and np.isnan(name)):
                name = appid
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


class Data:

    def user_game_matrix(self):
        ug = UserGame(self.dat)
        ug.create_matrix()
        return ug
        

class ReviewData(Data):
    def __init__(self) -> None:
        super().__init__()        
        data_files = sorted([os.path.join(DATA_DIR, fname) for fname in os.listdir(DATA_DIR) 
                            if fname.startswith('reviews') and fname.endswith('.csv')])

        self.trn_files = data_files[:5]
        
    def load(self):
        self.dat = pd.concat([pd.read_csv(fpath) for fpath in self.trn_files])

        appid_reader = AppIDReader()

        self.dat = self.dat.merge(appid_reader.app_names_dat, on='appid', how='left')
        missing_names_rowinds = self.dat['name'].isnull()
        self.dat.loc[missing_names_rowinds, 'name'] = self.dat.loc[missing_names_rowinds, 'appid'].astype(str)
        

class HoursPlayedData(Data):
    def __init__(self) -> None:
        super().__init__()

class UserGame:
    def __init__(self, dat: pd.DataFrame):
        self.dat = dat

    def create_matrix(self):
        """
        Generates a sparse matrix from ratings dataframe.

        Adapted from https://www.jillcates.com/pydata-workshop/html/tutorial.html.

        Args:
            df: pandas dataframe containing 3 columns (userId, gameId, rating)

        Returns:
            X: sparse matrix
            user_mapper: dict that maps user id's to user indices
            user_inv_mapper: dict that maps user indices to user id's
            game_mapper: dict that maps game id's to game indices
            game_inv_mapper: dict that maps game indices to game id's
        """
        M = self.dat['steamid'].nunique()
        N = self.dat['name'].nunique()

        self.user_mapper = dict(zip(np.unique(self.dat["steamid"]), list(range(M))))
        self.game_mapper = dict(zip(np.unique(self.dat["name"]), list(range(N))))

        self.user_inv_mapper = dict(zip(list(range(M)), np.unique(self.dat["steamid"])))
        self.game_inv_mapper = dict(zip(list(range(N)), np.unique(self.dat["name"])))

        user_index = [self.user_mapper[i] for i in self.dat['steamid']]
        item_index = [self.game_mapper[i] for i in self.dat['name']]

        self.X = csr_matrix((self.dat["voted_up"], (user_index, item_index)), shape=(M, N))

    @cached_property
    def sparsity(self):
        n_total = self.X.shape[0] * self.X.shape[1]
        n_ratings = self.X.nnz
        sparsity = n_ratings / n_total
        return sparsity

    @cached_property
    def ratings_per_game(self):
        return self.X.getnnz(axis=0)
    
    @cached_property
    def ratings_per_user(self):
        return self.X.getnnz(axis=1)

    def report_stats(self):
       

        print(f"% cells with a value: {self.sparsity:.2%}")
        
        print(f"Most active user rated {self.ratings_per_user.max()} games.")
        print(f"Least active user rated {self.ratings_per_user.min()} games.")

        print(f"Most rated game has {self.ratings_per_game.max()} ratings.")
        print(f"Least rated game has {self.ratings_per_game.min()} ratings.")

    @property
    def ratings_per_game_plot(self):
        return _make_rating_counts_plot(self.ratings_per_game) + pn.scale_x_log10(labels=lambda ar: [f"{int(k):,}" for k in ar])

    @property
    def ratings_per_user_plot(self):
        return _make_rating_counts_plot(self.ratings_per_user)

def _make_rating_counts_plot(rating_counts: np.ndarray, ):
    dat = pd.DataFrame(rating_counts, columns=['ratings'])
    return (
        pn.ggplot(dat, pn.aes(x='ratings')) +
        pn.geom_histogram() +
        pn.theme_bw() +
        pn.theme(figure_size=(3, 5))
    )