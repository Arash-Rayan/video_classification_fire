from scripts.train import train_and_evaluate
from models.conv2dplus1d import FireDetectorConv2D1D 
from models.conv3d_residual import FireDetectorWithResidual
from models.resnet3d import resnet_model
from models.slow_fast_model_train import slow_fast_model , train_and_evaluate as slow_fast_train
from utils.Dataset import data_loader
from utils.slow_fast_Dataset import data_loader as slow_data_loader
from utils.EvalMetrics import loss_fn
from configs.config import args
import matplotlib.pyplot as plt 
from utils.Plot import plot_performnace

loader = data_loader()
slow_loader = slow_data_loader()

available_models = {
    'conv2d1d' : FireDetectorConv2D1D() ,
    'conv3dResidual': FireDetectorWithResidual(),
    'resnet3d' : resnet_model, 
    'slowfast' : slow_fast_model
}

# train_losses, train_accuracies , train_recall, val_losses, val_accuracies , val_recall= train_and_evaluate(available_models['resnet3d'], 'resnet3d', loader['train_loader'], loader['test_loader'], loss_fn , args.epochs)
train_losses, train_accuracies , train_recall, val_losses, val_accuracies , val_recall= slow_fast_train(available_models['slowfast'] , 'slow_fast' ,slow_loader['train_loader'], slow_loader['test_loader'], loss_fn , args.epochs)


plot_performnace(len(train_losses), train_losses, val_losses ,model_name='conv2d1d' , save=True)


print(
       f'train loss is {train_losses[-1]:2f}\n',
       f'train accuracy is {train_accuracies[-1]:.2f}\n', 
       f'recall is {train_recall[-1]:.2f}\n'
       f'val loss is {val_losses[-1]:2f}\n',
       f'val accuracy is {val_accuracies[-1]:.2f}\n', 
       f'recall is {val_recall[-1]:.2f}\n', 
       )
