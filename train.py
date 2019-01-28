import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import argparse

from models.saliency_add import SaliencyAdd
from models.saliency_conv import SaliencyConv
from models.transform_cache import TransformCache
from models.cached_vgg16 import CachedVGG16
from models.scorer import LTR_score
from models.vgg16 import vgg16
from models.resnet import resnet152
from models.vip import ViP_features

from utils.cluewebDataset import ClueWeb12Dataset
from utils.evaluate import Evaluate

import tensorboard_logger as tfl
import logging
import copy
import os

from models.inception import inception_v3
from utils.vectorCache import VectorCache

FORMAT = '%(name)s: [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger("train")

torch.backends.cudnn.deterministic = True
torch.manual_seed(1337)


def pair_hinge_loss(positive, negative):
    loss = torch.clamp(1.0 - positive + negative, 0.0)
    return loss.mean()

"""
This method prepares the dataloaders for training and returns a training/validation dataloader.
"""
def prepare_dataloaders(train_file, test_file, vali_file):
    # Get the train/test datasets
    train_dataset = ClueWeb12Dataset(FLAGS.image_path, train_file, FLAGS.load_images,
                                     FLAGS.query_specific, FLAGS.only_with_image, FLAGS.size, FLAGS.grayscale,
                                     FLAGS.vector_cache, FLAGS.saliency_dir, saliency_cache=FLAGS.saliency_cache)
    test_dataset = ClueWeb12Dataset(FLAGS.image_path, test_file, FLAGS.load_images,
                                    FLAGS.query_specific, FLAGS.only_with_image, FLAGS.size, FLAGS.grayscale,
                                    FLAGS.vector_cache, FLAGS.saliency_dir, saliency_cache=FLAGS.saliency_cache)
    vali_dataset = ClueWeb12Dataset(FLAGS.image_path, vali_file, FLAGS.load_images,
                                    FLAGS.query_specific, FLAGS.only_with_image, FLAGS.size, FLAGS.grayscale,
                                    FLAGS.vector_cache, FLAGS.saliency_dir, saliency_cache=FLAGS.saliency_cache)

    # Prepare the loaders
    # note that with h5py only one worker can be used to access the dataset.
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                  shuffle=True, num_workers=1)
    # Initiate the Evaluation classes
    trainEval = Evaluate(train_file, train_dataset, FLAGS.load_images, "train", vector_cache=FLAGS.vector_cache,
                         batch_size=FLAGS.batch_size)
    testEval = Evaluate(test_file, test_dataset, FLAGS.load_images, "test", vector_cache=FLAGS.vector_cache,
                        batch_size=FLAGS.batch_size)
    valiEval = Evaluate(vali_file, vali_dataset, FLAGS.load_images, "validation", vector_cache=FLAGS.vector_cache,
                        batch_size=FLAGS.batch_size)

    return dataloader, trainEval, testEval, valiEval


"""
The fold iterator provides files to train, test and validate on for all folds and sessions.
To generator yields a test, train and validate file path that should be used for the current model.
"""
def fold_iterator():
    for fold in range(1, FLAGS.folds+1):
        for session in range(FLAGS.sessions_per_fold):
            fold_path = os.path.join(FLAGS.content_feature_dir, "Fold{}".format(fold))
            test = os.path.join(fold_path, "test.txt")
            train = os.path.join(fold_path, "train.txt")
            vali = os.path.join(fold_path, "vali.txt")
            yield train, test, vali



def train_model(model, criterion, dataloader, trainEval, testEval,
                use_gpu, optimizer, scheduler, description, num_epochs=25):
    
    tf_logger = tfl.Logger(FLAGS.log_dir.format(description))

    # Set model to training mode
    model.train(False)  
    best_model = copy.deepcopy(model)
    trainEval.eval(model, tf_logger, 0)
    test_scores = testEval.eval(model, tf_logger, 0)
    best_test_score = test_scores

    for epoch in range(num_epochs + 1):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

         
        model.train(True)  
        # Each epoch has a training and validation phase
        if scheduler is not None:
            scheduler.step()

        running_loss = 0.0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and wrap them into varaibles
            # TODO, can we move the variable loading to the dataloader itself?
            if use_gpu:
                p_static_features = Variable(data[0][1].float().cuda())
                n_static_features = Variable(data[1][1].float().cuda())
                if FLAGS.load_images:
                    p_image = Variable(data[0][0].float().cuda())
                    n_image = Variable(data[1][0].float().cuda())
                if FLAGS.saliency_dir:
                    p_saliency = Variable(data[0][3].float().cuda())
                    n_saliency = Variable(data[1][3].float().cuda())
            else:
                p_static_features = Variable(data[0][1].float())
                n_static_features = Variable(data[1][1].float())
                if FLAGS.load_images:
                    p_image = Variable(data[0][0].float())
                    n_image = Variable(data[1][0].float())
                if FLAGS.saliency_dir:
                    p_saliency = Variable(data[0][3].float())
                    n_saliency = Variable(data[1][3].float())

            if not FLAGS.load_images:
                p_image = n_image = None

            if not FLAGS.saliency_dir:
                p_saliency = n_saliency = None

            positive = model.forward(p_image, p_static_features, p_saliency)
            negative = model.forward(n_image, n_static_features, n_saliency)

            # Compute the loss
            loss = criterion(positive, negative)

            running_loss += loss.data[0].item() # * p_static_features.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.train(False) 
        trainEval.eval(model, tf_logger, epoch)
        test_scores = testEval.eval(model, tf_logger, epoch)
        if best_test_score[FLAGS.optimize_on] < test_scores[FLAGS.optimize_on]:
            logger.debug("Improved the current best score.")
            best_test_score = test_scores
            best_model = copy.deepcopy(model)

        tf_logger.log_value('train_loss', running_loss, epoch)
        logger.info('Train_loss: {}'.format(running_loss))

    return best_test_score, best_model

"""
Prepare the model with the correct weights and format the the configured use.
"""
def prepare_model(use_scheduler=True):
    use_gpu = torch.cuda.is_available()

    # TODO THIS SHOULD BE REFACTORED
    if FLAGS.model == "ViP":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size, ViP_features(16, 10, FLAGS.batch_size))
    elif FLAGS.model == "vgg16":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size, vgg16(pretrained=True, state_dict=None, output_size=FLAGS.visual_features))
        for param in model.feature_model.features.parameters():
            param.requires_grad = False
    elif FLAGS.model == "resnet152":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size, resnet152(pretrained=True, state_dict=None, output_size=FLAGS.visual_features))
        for param in list(model.feature_model.parameters())[:-FLAGS.finetune_n_layers]:
            param.requires_grad = False
    elif FLAGS.model == "resnet18":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size, resnet152(pretrained=True, state_dict=None, output_size=FLAGS.visual_features))
        for param in list(model.feature_model.parameters())[:-FLAGS.finetune_n_layers]:
            param.requires_grad = False
    elif FLAGS.model == "inception":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size, inception_v3(pretrained=True, output_size=FLAGS.visual_features))
        for param in list(model.feature_model.parameters())[:-FLAGS.finetune_n_layers]:
            param.requires_grad = False
    elif FLAGS.model == "features_only":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size)
    elif FLAGS.model == "cached_vgg16":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size, CachedVGG16(output_size=FLAGS.visual_features))
    elif FLAGS.model == "transform_cache":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size,
                          TransformCache(input_size=FLAGS.cache_vector_size, hidden_layers=FLAGS.visual_layers,
                                         output_size=FLAGS.visual_features, dropout=FLAGS.visual_dropout))
    elif FLAGS.model == "saliency_add": # TODO integrate this with a separate flag.
        visual_model = TransformCache(input_size=FLAGS.cache_vector_size, hidden_layers=FLAGS.visual_layers,
                                         output_size=FLAGS.visual_features, dropout=FLAGS.visual_dropout)
        saliency_model = SaliencyConv()
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size,
                          SaliencyAdd(visual_model, saliency_model,
                                      hidden_layers='4096x4096',
                                      output_size=FLAGS.visual_features, dropout=FLAGS.visual_dropout))
    elif FLAGS.model == "saliency_twin_add": # TODO integrate this with a separate flag.
        visual_model = TransformCache(input_size=FLAGS.cache_vector_size, hidden_layers=FLAGS.visual_layers,
                                         output_size=FLAGS.visual_features, dropout=FLAGS.visual_dropout)
        saliency_model = TransformCache(input_size=FLAGS.cache_vector_size, hidden_layers=FLAGS.visual_layers,
                                         output_size=FLAGS.visual_features, dropout=FLAGS.visual_dropout)
        model = LTR_score(FLAGS.content_feature_size, FLAGS.classification_dropout, FLAGS.hidden_size,
                          SaliencyAdd(visual_model, saliency_model,
                                      hidden_layers='4096x4096',
                                      output_size=FLAGS.visual_features, dropout=FLAGS.visual_dropout))
    else:
        raise NotImplementedError("Model: {} is not implemented".format(FLAGS.model))

    if use_gpu:
        model = model.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    logger.info("Total model parameters: {}".format(count_parameters(model)))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FLAGS.learning_rate, weight_decay=1e-5)

    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, optimizer, scheduler, use_gpu


def train():
    test_scores = {}
    vali_scores = {}
    for i, (train, test, vali) in enumerate(fold_iterator(), 1):
        # Prepare all model components and initalize parameters.
        model, optimizer, scheduler, use_gpu = prepare_model()

        # Create a dataloader for training and three evaluation classes.
        dataloader, trainEval, testEval, valiEval = prepare_dataloaders(train, test, vali)

        if i == 1:
            logger.info(model)

        description = FLAGS.description + "_" + str(i)
        test_score, model = train_model(model, pair_hinge_loss, dataloader, trainEval, testEval,
                                        use_gpu, optimizer, scheduler, description, num_epochs=FLAGS.epochs)

        # Add and store the newly added scores.
        vali_score, rank_df = valiEval.eval(model, get_df=True)
        
        # Store the results to an excel sheet.
        # writer = pd.ExcelWriter("{}_{}_{}.xlsx".format(FLAGS.optimized_scores_path, description, vali_score[FLAGS.optimize_on]))
        rank_df.to_pickle("{}_{}.pkl".format(FLAGS.optimized_scores_path, description))
        # writer.save()

        # Store the scores to list.
        test_scores = testEval.add_scores(test_scores, test_score)
        vali_scores = valiEval.add_scores(vali_scores, vali_score)
        testEval.store_scores(FLAGS.optimized_scores_path + FLAGS.description, description, test_score)
        valiEval.store_scores(FLAGS.optimized_scores_path + FLAGS.description, description, vali_score)

    # Average the test and validation scores.
    test_scores = testEval.avg_scores(test_scores, i)
    vali_scores = valiEval.avg_scores(vali_scores, i)

    logger.info("Finished, printing best results now.")
    testEval.print_scores(test_scores)
    valiEval.print_scores(vali_scores)
    testEval.store_scores(FLAGS.optimized_scores_path + FLAGS.description, FLAGS.description + "_final", test_scores)
    valiEval.store_scores(FLAGS.optimized_scores_path + FLAGS.description, FLAGS.description + "_final", vali_scores,
                          final=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_feature_dir', type=str, default='storage/clueweb12_web_trec/',
                        help='The location of all the folds with train, test and validation files.')
    parser.add_argument('--folds', type=int, default=5,
                        help='The amounts of folds to train on.')
    parser.add_argument('--sessions_per_fold', type=int, default=1,
                        help='The amount of training sessions to average per fold.')
    parser.add_argument('--image_path', type=str, default='storage/images/snapshots/',
                        help='The location of the salicon images for training.')
    parser.add_argument('--saliency_dir', type=str, default=None,
                        help='[optional] The path of the directory where the saliency images are stored. No saliency '
                             'images will be stored the argument is not passed')
    parser.add_argument('--saliency_cache_path', type=str, default=None,
                        help='[optional] The path of the directory where the saliency images cache is stored. ie.'
                             'storage/model_cache/restnet152-saliency-cache.')
    parser.add_argument('--cache_path', type=str, default=None,
                        help='Provide the path of a feature extractor cache path in order to speed up training ie. '
                             'storage/model_cache/restnet152-saliency-cache. ')

    parser.add_argument('--batch_size', type=int, default=3,
                        help='The batch size used for training.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The amount of epochs used to train.')
    parser.add_argument('--description', type=str, default='example_run',
                        help='The description of the run, for logging, output and weights naming.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate to use for the experiment')
    parser.add_argument('--content_feature_size', type=int, default=11,
                        help='The amount of context features')
    parser.add_argument('--model', type=str, default="features_only",
                        help='chose the model to train, (features_only, ViP)')
    parser.add_argument('--load_images', type=str, default="True",
                        help='set whether the images should be loaded during training and evaluation.')
    parser.add_argument('--only_with_image', type=str, default="True",
                        help='set whether all documents without images should be excluded from the dataset')
    parser.add_argument('--query_specific', type=str, default="False",
                        help='set whether the images are query specific (ie. using query specific highlights)')
    parser.add_argument('--log_dir', type=str, default='storage/logs/{}',
                        help='The location to place the tensorboard logs.')
    parser.add_argument('--optimized_scores_path', type=str, default='storage/logs/',
                        help='The location to store the scores that were optimized.')
    parser.add_argument('--optimize_on', type=str, default='ndcg@5',
                        help='Give the measure to optimize the model on (ndcg@1, ndcg@5, ndcg@10, p@1, p@5, p@10, map).')
    parser.add_argument('--grayscale', type=str, default='False',
                        help='Flag whether to convert the images to grayscale.')

    parser.add_argument('--classification_dropout', type=float, default=.1,
                        help='The dropout to use in the classification layer.')
    parser.add_argument('--visual_dropout', type=float, default=.1,
                        help='The dropout to use in the visual feature layer.')
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='The amount of hidden layers in the classification layer')
    parser.add_argument('--visual_layers', type=str, default="2048x2048",
                        help="[cached only] Provide hidden sizes seperated by an 'x' that transforms the visual outputs to a hidden" 
                             "representation")
    parser.add_argument('--cache_vector_size', type=int, default=25088,
                        help="[cached only] the size of the output vectors stored in the cache")
    parser.add_argument('--visual_features', type=int, default=30,
                        help='The size of the visual feature vector')
    parser.add_argument('--finetune_n_layers', type=int, default=1,
                        help='For resnet152 and inception, define the amount of layers at the end to be fine tuned. ')


    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.load_images = FLAGS.load_images == "True"
    FLAGS.only_with_image = FLAGS.only_with_image == "True"
    FLAGS.query_specific = FLAGS.query_specific == "True"
    FLAGS.grayscale = FLAGS.grayscale == "True"

    if FLAGS.model in ("vgg16", "resnet152", "resnet18"):
        FLAGS.size = (224,224)
    if FLAGS.model in ("inception"):
        FLAGS.size = (299,299)
    else:
        FLAGS.size = (64,64)

    if FLAGS.cache_path is not None:
        FLAGS.vector_cache = VectorCache(cache_path=FLAGS.cache_path)
    else:
        FLAGS.vector_cache = None

    if FLAGS.saliency_cache_path is not None:
        FLAGS.saliency_cache = VectorCache(cache_path=FLAGS.saliency_cache_path)
    else:
        FLAGS.saliency_cache = None

    logger.info(FLAGS)

    train()

