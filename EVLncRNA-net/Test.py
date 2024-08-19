import config
from train import *
from train import test
if __name__ == '__main__':
    if config.data_name == 'human':
            model = mynet( label_num=2, K=3, d=3, node_hidden_dim=3, other_feature_dim=128,  other_feature_dim_in=2000,pnode_nn=False, fnode_nn=False ).to( device ) #human_net
    else:
            model = mynet( label_num=2, K=3, d=3, node_hidden_dim=3, other_feature_dim=128, other_feature_dim_in=4000,data_name=config.data_name ).to( device )  # mouse & plant
    
    human_path = 'mode/human.ckpt'
    mouse_path = 'model/mouse.ckpt'
    plant_path = 'model/plant.ckpt'
    
    if config.data_name == 'human':
        model_path = human_path
    elif config.data_name == 'mouse':
        model_path = mouse_path
    elif config.data_name == 'plant':
        model_path = plant_path
    else:
        model_path = 'default/path.ckpt'  # 默认路径
    
    print(model_path)
    test(model=model,fasta_file=config.test_x,label_file=config.test_label,feature_file=config.test_fea,model_path=model_path) #human_tset