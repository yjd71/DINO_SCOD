import os
import torch


class Config():
    def __init__(self):
        # data path
        self.dataset_dir = './Dataset/COD'
        
        ''' Train Dataset: CAMO-Train + COD10K-Train '''
        self.train_sets = ['TR-CAMO', 'TR-COD10K']
        self.train_imgs = [os.path.join(self.dataset_dir, name, 'im') for name in self.train_sets]
        self.train_masks = [os.path.join(self.dataset_dir, name, 'gt') for name in self.train_sets]
        self.train_sample_txt = os.path.join(self.dataset_dir, 'sampled_images.txt')
        self.train_labeled_indices_pt = None
        
        ''' Test Dataset '''
        # CHAMELEON
        self.test_CHAMELEON_imgs = os.path.join(self.dataset_dir, 'CHAMELEON', 'im')
        self.test_CHAMELEON_masks = os.path.join(self.dataset_dir, 'CHAMELEON', 'gt')
        # CAMO-Test
        self.test_CAMO_imgs = os.path.join(self.dataset_dir, 'TE-CAMO', 'im')
        self.test_CAMO_masks = os.path.join(self.dataset_dir, 'TE-CAMO', 'gt')
        # COD10K-Test
        self.test_COD10K_imgs = os.path.join(self.dataset_dir, 'TE-COD10K', 'im')
        self.test_COD10K_masks = os.path.join(self.dataset_dir, 'TE-COD10K', 'gt')
        # NC4K
        self.test_NC4K_imgs = os.path.join(self.dataset_dir, 'NC4K', 'im')
        self.test_NC4K_masks = os.path.join(self.dataset_dir, 'NC4K', 'gt')


        self.num_workers = 8

        self.train_size = 392
        self.test_size = 392

        ''' Save Paths '''
        self.result_path = './results'
        self.dir_name = 'base_model'
        self.save_dir = os.path.join(self.result_path, self.dir_name)

        self.CUDA = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        self.epochs = 30
        self.batch_size = 16

        ''' Data Augmentation '''
        self.rVFlip = True
        self.rCrop = True
        self.rRotate = False
        self.colorEnhance = True
        self.rPeper = False

        ''' Optimizer '''
        self.weight_decay = 0

        ''' LR Scheduler '''
        self.learning_rate = 1e-4
        self.min_lr = 1e-7

        ''' logging '''
        self.log_interval = 20


if __name__ == '__main__':
    config = Config()
    print(config)
