import dill

class container:
    def __init__(self,_output, _Config):
        self.output = _output
        self.config = _Config
    
    def change_output(self, _output):
        self.output = _output
    
    def weights(self, _class_weights):
        self.class_weights = _class_weights
        
    def loaders(self, train, val, test):
        self.train_loader = train
        self.val_loader = val
        self.test_loader = test
    
    def model(self, _model_last, _model_best):
        self.model_last = _model_last
        self.model_best = _model_best
    
    def add_device(self, _device):
        self.device = _device
        
    def add_true_pred(self,true_array,pred_array,percentage_array):
        self.true = true_array
        self.pred = pred_array
        self.perc = percentage_array
        
    def dump(self, file_name):
        with open(file_name, "wb") as f:
            dill.dump(self, f)
        # my_container = Container1
        # my_container.dump("my_container.pickle")
        
    def load(self, file_name):
        with open(file_name, "rb") as f:
            loaded_obj = dill.load(f)
            self.__dict__.update(loaded_obj.__dict__)
        # loaded_container = container(None, None)
        # loaded_container.load("my_container.pickle")