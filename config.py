import os
import os.path as path

class Config():
    def __init__(self, dataset, model_home, do_ensemble=False):
        self.data_home = path.join("dataset", dataset)
        self.model_home = model_home
        self.labeled = "labeled_data.csv"
        self.test = "test_data.csv"
        self.unlabeled = "unlabeled_data.csv"
        self.retrain = False
        self.new_samples = 30000
        
        if do_ensemble:
            self.ensemble_home = path.join(self.model_home, "emsemble")
            if not path.isdir(self.ensemble_home):
                os.mkdir(self.ensemble_home)
            
            self.cache_path = self.ensemble_home
            self.overwrite_cache = False