import os.path
import torch
from torchvision.io.image import read_image
#通过修改self.data_name来修改不同的数据集，只需要改这一个参数就好了

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000  # 若超过2000batch效果还没提升，则提前结束训练
        self.num_classes =2  # 类别数无需修改
        self.num_epochs = 20  # epoch数
        self.batch_size =32  # mini-batch大小，看显存大小和数据集大小决定 老鼠16 植物数据集建议为16 人类为
        self.learning_rate = 1e-3 #学习率   植物老鼠为1e-3

        self.data_name='human'#数据种类 三种可选 human mouse plant
        self.train_x='dataset/'+self.data_name+'/'+self.data_name+'_train.fa'
        self.train_fea='dataset/'+self.data_name+'/'+'train_fea.npz'
        self.train_label='dataset/'+self.data_name+'/'+self.data_name+'_train_label'#验证集数据

        self.test_x='dataset/'+self.data_name+'/'+self.data_name+'_test.fa'
        self.test_fea='dataset/'+self.data_name+'/'+'test_fea.npz'
        self.test_label='dataset/'+self.data_name+'/'+self.data_name+'_test_label'#验证集数据

        if not os.path.exists('model'):
            os.makedirs('model')


        self.save_path = 'model/'+self.data_name+'.ckpt'#保存模型的路径
        self.log_dir= './log/'+self.data_name#tensorboard日志的路径

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


