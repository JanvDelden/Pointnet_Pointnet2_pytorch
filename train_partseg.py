"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
import custom_functions.transform as t
import provider


from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Tree': [0, 1]}
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--weight', type=float, default=5.2, help='weight to be applied to loss of tree points')
    parser.add_argument('--adaptive', action='store_true', default=False, help='use adaptive loss weights')

    return parser.parse_args()


def main(args):

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    performance_dir = exp_dir.joinpath('performance/')
    performance_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DEFINE DEVICE FOR TRAINING AND OTHER DATA RELATED STUFF'''
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    if use_cuda:
        # define paths where split information is located for dataset
        root = '/content/Pointnet_Pointnet2_pytorch/data'
        trainpath = "/trainsplit.npy"
        testpath = "/valsplit.npy"

        # save split information for later purposes
        trainsplit = np.load(root + trainpath)
        testsplit = np.load(root + testpath)

        split_dir = exp_dir.joinpath('split/')
        split_dir.mkdir(exist_ok=True)
        trainsplit_path = str(split_dir) + trainpath
        testsplit_path = str(split_dir) + testpath

        np.save(trainsplit_path, trainsplit)
        np.save(testsplit_path, testsplit)
    else:
        #root = 'C:/Users/Jan Schneider/OneDrive/Studium/statistisches Praktikum/treelearning/data/tmp'
        root = "G:/Meine Ablage/Colab/tree_learning/data/chunks"
        trainpath = "/trainsplit.npy"
        testpath = "/valsplit.npy"

    '''TRANSFORMATIONS TO BE APPLIED DURING TRAINING AND TEST TIME'''
    traintransform = t.Compose([t.Normalize(),
                                t.RandomScale(anisotropic=True, scale=[0.8, 1.2]),
                                t.RandomRotate(),
                                t.RandomFlip(),
                                t.RandomJitter()])
    testtransform = t.Compose([t.Normalize()])

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, transform=traintransform, splitpath=root + trainpath, normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, transform=testtransform, splitpath=root + testpath, normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 1
    num_parts = 2

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    weights = torch.tensor([1, args.weight])
    weights = weights.to(device)
    weights = weights.float()

    classifier = MODEL.get_model(num_parts, num_classes, normal_channel=args.normal).to(device)
    criterion = MODEL.get_loss(weights=weights, batch_size=args.batch_size, adaptive=args.adaptive, device=device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')

        # train metrics
        train_accs = checkpoint['train_accs']
        train_w_accs = checkpoint['train_w_accs']
        train_loss = checkpoint['train_loss']
        train_miou = checkpoint['train_miou']

        # val metrics
        val_accs = checkpoint['val_accs']
        val_w_accs = checkpoint['val_w_accs']
        val_loss = checkpoint['val_loss']
        val_miou = checkpoint['val_miou']


    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

        # train metrics
        train_accs = []
        train_w_accs = []
        train_tree_accs = []
        train_no_tree_accs = []
        train_loss = []
        train_miou = []

        # val metrics
        val_accs = []
        val_w_accs = []
        val_tree_accs = []
        val_no_tree_accs = []
        val_loss = []
        val_miou = []



    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0 # todo
    best_inctance_avg_iou = 0

    '''
    START TRAINING
    '''

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        mean_loss = []
        w_acc = []
        acc_tree = []
        acc_no_tree = []
        miou = []



        '''adjust training parameters'''
        log_string('\nEpoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        # print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points, target, n_sampled_points = provider.random_point_dropout(points, target) # this is different from test
            NUM_POINT = points.size()[1]
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_parts)
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, n_sampled_points)
            mean_loss.append(loss.item())

            loss.backward() # this is different from test
            optimizer.step() # this is different from test

            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * NUM_POINT))

            pred_choice = pred_choice.cpu().numpy()
            target = target.cpu().data.numpy()

            # weighted accuracy for batch
            tp = np.sum(np.logical_and(target == 1, pred_choice == 1))
            fn = np.sum(np.logical_and(target == 1, pred_choice == 0))
            fp = np.sum(np.logical_and(target == 0, pred_choice == 1))
            tn = np.sum(np.logical_and(target == 0, pred_choice == 0))

            acc_tree.append(tp / (tp + fn))
            acc_no_tree.append(tn / (tn + fp))
            w_acc.append((tp / (tp + fn) + tn / (tn + fp)) / 2)

            # iou for batch
            iou_tree = tp / (tp + fp + fn)
            iou_not_tree = tn / (tn + fn + fp)
            miou.append((iou_tree + iou_not_tree)/2)



        '''After one epoch, metrics aggregated over iterations'''
        train_accs.append(np.round(np.mean(mean_correct), 5))
        train_w_accs.append(np.round(np.mean(w_acc), 5))
        train_tree_accs.append(np.round(np.mean(acc_tree), 5))
        train_no_tree_accs.append(np.round(np.mean(acc_no_tree), 5))
        train_loss.append(np.round(np.mean(mean_loss), 5))
        train_miou.append(np.round(np.mean(miou), 5))

        log_string('Epoch %d trainloss: %f, trainacc: %f, trainwacc: %f, testacctree: %f, testaccnotree: %f, mIOU: %f' % (
            epoch + 1, train_loss[epoch], train_accs[epoch], train_w_accs[epoch], train_tree_accs[epoch], train_no_tree_accs[epoch], train_miou[epoch]
        ))



        '''validation set'''
        with torch.no_grad():
            mean_loss = []
            w_acc = []
            acc_tree = []
            acc_no_tree = []
            miou = []

            classifier = classifier.eval()

            '''apply current model to validation set'''
            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

                NUM_POINT = points.size()[1]
                points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
                points = points.transpose(2, 1)
                seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
                seg_pred = seg_pred.contiguous().view(-1, num_parts)
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, args.npoint)
                mean_loss.append(loss.item())

                pred_choice = seg_pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                mean_correct.append(correct.item() / (args.batch_size * NUM_POINT))

                pred_choice = pred_choice.cpu().numpy()
                target = target.cpu().data.numpy()

                # weighted accuracy for batch
                tp = np.sum(np.logical_and(target == 1, pred_choice == 1))
                fn = np.sum(np.logical_and(target == 1, pred_choice == 0))
                fp = np.sum(np.logical_and(target == 0, pred_choice == 1))
                tn = np.sum(np.logical_and(target == 0, pred_choice == 0))

                acc_tree.append(tp / (tp + fn))
                acc_no_tree.append(tn / (tn + fp))
                w_acc.append((acc_tree[batch_id] + acc_no_tree[batch_id]) / 2)

                # iou for batch
                iou_tree = tp / (tp + fp + fn)
                iou_not_tree = tn / (tn + fn + fp)
                miou.append((iou_tree + iou_not_tree) / 2)


        '''After one epoch, metrics aggregated over iterations'''
        val_accs.append(np.round(np.mean(mean_correct), 5))
        val_w_accs.append(np.round(np.mean(w_acc), 5))
        val_tree_accs.append(np.round(np.mean(acc_tree), 5))
        val_no_tree_accs.append(np.round(np.mean(acc_no_tree), 5))
        val_loss.append(np.round(np.mean(mean_loss), 5))
        val_miou.append(np.round(np.mean(miou), 5))

        log_string('Epoch %d testloss: %f, testacc: %f, testwacc: %f, testacctree: %f, testaccnotree: %f, testmIOU: %f' % (
            epoch + 1,val_loss[epoch], val_accs[epoch], val_w_accs[epoch], val_tree_accs[epoch], val_no_tree_accs[epoch], val_miou[epoch]
        ))

        if val_w_accs[epoch] >= np.max(val_w_accs):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch+1,
                'train_accs': train_accs,
                'train_loss': train_loss,
                'train_w_accs': train_w_accs,
                'train_miou': train_miou,
                'val_accs': val_accs,
                'val_w_accs': val_w_accs,
                'val_miou': val_miou,
                'val_loss': val_loss,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        global_epoch += 1

        # if test_metrics['accuracy'] > best_acc:
        #     best_acc = test_metrics['accuracy']
        # if test_metrics['class_avg_iou'] > best_class_avg_iou:
        #     best_class_avg_iou = test_metrics['class_avg_iou']
        # if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou: # todo
        #     best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        # log_string('Best accuracy is: %.5f' % best_acc)
        # log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        # log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)

    # save performance measures
    accs_path = str(performance_dir) + '/accs.npy'
    w_accs_path = str(performance_dir) + '/w_accs.npy'
    tree_accs_path = str(performance_dir) + "/tree_accs.npy"
    no_tree_accs_path = str(performance_dir) + "/no_tree_accs.npy"
    loss_path = str(performance_dir) + '/loss.npy'
    mious_path = str(performance_dir) + '/mious.npy'

    accs = np.array([train_accs, val_accs]).T
    w_accs = np.array([train_w_accs, val_w_accs]).T
    tree_accs = np.array([train_tree_accs, val_tree_accs]).T
    no_tree_accs = np.array([train_no_tree_accs, val_no_tree_accs]).T
    loss = np.array([train_loss, val_loss]).T
    mious = np.array([train_miou, val_miou]).T

    np.save(accs_path, accs)
    np.save(w_accs_path, w_accs)
    np.save(tree_accs_path, tree_accs)
    np.save(no_tree_accs_path, no_tree_accs)
    np.save(loss_path, loss)
    np.save(mious_path, mious)


if __name__ == '__main__':
    args = parse_args()
    main(args)
