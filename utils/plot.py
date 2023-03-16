import matplotlib.pyplot as plt
import os


def loss_plot(args, fold, train_x, train_y, valid_x, valid_y):
    num = args.epochs
    plot_save_path = args.plots_dir
    x = [i for i in range(num)]
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    filename = str(args.arch) + '_' + str(args.epochs) + '_' + str(args.batch_size) + '_' + str(fold) + '_loss.jpg'
    save_loss = os.path.join(plot_save_path, filename)
    plt.figure()
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')
    plt.legend()
    plt.savefig(save_loss)


def metrics_plot(arg, fold, name, *args):
    num = arg.epochs
    names = name.split('&')
    plot_save_path = arg.plots_dir
    metrics_value = args
    i = 0
    x = [i for i in range(num)]
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    filename = str(arg.arch) + '_' + str(arg.epochs) + '_' + str(arg.batch_size) + '_' + str(fold)  + '_' + name + '.jpg'
    save_metrics = os.path.join(plot_save_path, filename)
    plt.figure()
    for l in metrics_value:
        plt.plot(x, l, label=str(names[i]))
        # plt.scatter(x,l,label=str(l))
        i += 1
    plt.legend()
    plt.savefig(save_metrics)
