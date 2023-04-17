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
    
    def start_model(self, _model):
        self.model = _model
        
    def end_model(self, _model):
        self.model = _model
    
    def add_device(self, _device):
        self.device = _device