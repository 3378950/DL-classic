from train import train_model
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DML Training')
    parser.add_argument('--model', help='model', default="VGG", type=str)
    parser.add_argument('--in_dim', help='dimonsion of input', default=1, type=int)
    parser.add_argument('--classes', help='number of classes', default=10, type=int)
    parser.add_argument('--optim', help='optimizer', default="SGD", type=str)
    parser.add_argument('--loss_fn', help='loss function', default="cross entropy", type=str)
    parser.add_argument('--batch_size', help='Batch_size', default=16, type=int)
    parser.add_argument('--n_epochs', help='Number of Epochs', default=3,  type=int)
    parser.add_argument('--learning_rate', help='Learning Rate', default=1e-2, type=float)
    parser.add_argument('--saved_epoch', help='epoch of saved model', default=None, type=int)
    parser.add_argument('--device', help='name of device', default="cuda:0", type=str)
    args = parser.parse_args()
    return args

args = parse_args()
train_model(
    model=args.model,
    in_dim=args.in_dim,
    classes=args.classes,
    optim_type=args.optim,
    loss_fn=args.loss_fn,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    learning_rate=args.learning_rate,
    saved_epoch=args.saved_epoch,
    device_name=args.device,
    )