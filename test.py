from torch.utils.data import DataLoader, RandomSampler
import torchvision.transforms as transforms
import time
import torch
import os
import numpy as np
from dataset import split_dataset,add_predefined_trigger_cifar,add_trigger_cifar,generate_trigger
from batchnorm import transfer_bn_to_noisy_bn
from anp_utils import mask_train,clip_mask,save_mask_scores,test,reset,evaluate_by_number,evaluate_by_threshold,read_data
from torchvision.datasets import CIFAR10
'''
下面这些参数是需要调整的超参数
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
trigger_info = None
data_dir = "./data"
output_dir = "./data"
val_frac=0.01
batch_size=128
print_every=500
nb_iter = 2000
anp_eps=0.4
anp_steps=1
anp_alpha=0.2
# pruning_by='threshold'
pruning_by='number'
pruning_max=0.90
pruning_step=0.05
def main():

    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.

    orig_train = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    _, clean_val = split_dataset(dataset=orig_train, val_frac=val_frac)
    clean_test = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    poison_test = add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)

    random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples=print_every * batch_size)
    clean_val_loader = DataLoader(clean_val, batch_size=batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0)
    poison_test_loader = DataLoader(poison_test, batch_size=batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=batch_size, num_workers=0)

    # Step 2: load model checkpoints and trigger info

    net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

    net = transfer_bn_to_noisy_bn(net)
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=0.2, momentum=0.9)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=anp_eps / anp_steps)

    # Step 3: train backdoored models
    # print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    # nb_repeat = int(np.ceil(nb_iter / print_every))
    # for i in range(nb_repeat):
    #     start = time.time()
    #     lr = mask_optimizer.param_groups[0]['lr']
    #     train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=clean_val_loader,
    #                                        mask_opt=mask_optimizer, noise_opt=noise_optimizer,device=device,anp_steps=anp_steps,anp_eps=anp_eps,anp_alpha=anp_alpha)
    #     cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader,device=device)
    #     po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader,device=device)
    #     end = time.time()
    #     print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
    #         (i + 1) * print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
    #         cl_test_loss, cl_test_acc))
    # save_mask_scores(net.state_dict(), os.path.join(output_dir, 'mask_values.txt'))


    # step 4 : pruning
    mask_file = os.path.join(output_dir, 'mask_values.txt')
    mask_values = read_data(mask_file)
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader,device=device)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader,device=device)
    print('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    if pruning_by == 'threshold':
        results = evaluate_by_threshold(
            net, mask_values, pruning_max=pruning_max, pruning_step=pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader,device=device
        )
    else:
        results = evaluate_by_number(
            net, mask_values, pruning_max=pruning_max, pruning_step=pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader,device=device
        )
    file_name = os.path.join(output_dir, 'pruning_by_{}.txt'.format(pruning_by))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        f.writelines(results)



if __name__ == '__main__':
    main()
