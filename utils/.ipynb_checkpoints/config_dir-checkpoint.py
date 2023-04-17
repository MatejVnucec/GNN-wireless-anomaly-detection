Config = {   
    "dataset_paths":{
        "TSSB":{
            "path_main":"datasets/TSSB/"
        },
        "Rutgers":{
            "path_main": "datasets/dataset_uncut.csv", #"dataset_uncut.csv", "dataset_cut.csv", "dataset_rss.npz"
            "path_properties": "datasets/dataset_properties.npz",
            "path_mask": "datasets/dataset_mask.npz",
            "len_type": "un/cut" # un/cut, random
        }
    },
    
    "graph":{
        "classif":"node", #"node","graph"
        "type":"VG", #"MTF", "VG", "Dual_VG"
        "MTF":{
            "num_bins":"auto" #you can imput an intiger or "auto" in which case it will chose the length of X as number of bins
        },
        "VG":{
            "edge_type": "natural", #"natural", "horizontal"
            "distance": 'distance', #'slope', 'abs_slope','distance','h_distance','v_distance','abs_v_distance',
            "edge_dir": "directed" #"undirected", "directed"
        },
        "stride_param":{
            "stride_type": "notflexible_with_remainder", #"flexible", "notflexible", "notflexible_with_remainder"
        }
    },
        
    "main":{
        "SEED": 300,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "range_epoch": 3000, #set length of epoch
        "save_file": " test_test",
        "name_of_save": "VG_Rutger",
        "patience": 500,
        "loss": "CE" #"BCE", "CE"
    },
    "train/val/test":{
        "train":0.8,
        "val":0.2, #of train
        "test":0.2
    },
    "keep_last": True
}