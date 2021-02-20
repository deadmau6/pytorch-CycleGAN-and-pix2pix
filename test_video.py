"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from cv2 import *

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    """
    # Source: https://github.com/Vinno97/realtime-pytorch-CycleGAN-and-pix2pix/blob/master/test.py
    """

    data = next(iter(dataset), None)

    if os.path.isfile(opt.videosource):
        src = os.path.abspath(opt.videosource)
    else:
        src = int(opt.videosource)

    webcam = VideoCapture(src)
    save_only = False
    size = (512, 512)
    out = None
    if opt.save and os.path.isfile(opt.videosource):
        #
        fpath = os.path.abspath(opt.videosource)
        fname = os.path.basename(fpath).split('.')[0]
        out_path = os.path.join(os.path.dirname(fpath), f'{fname}_style.mov')
        #
        size = (int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = webcam.get(cv2.CAP_PROP_FPS)
        out = VideoWriter(out_path, cv2.VideoWriter_fourcc(*'H264'), fps, size)
        save_only = True
    #
    if not save_only:
        namedWindow("cam-input")
        namedWindow("cam-output")
    #
    while True:
        success, input_image = webcam.read()
        if not success:
            print("Could not get an image. Please check your video source")
            break
        if not save_only:
            imshow("cam-input", input_image)

        input_image = cv2.resize(input_image, (256, 256))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.asarray([input_image])
        input_image = np.transpose(input_image, (0, 3, 1, 2))

        data['A'] = torch.FloatTensor(input_image)

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        result_image = model.get_current_visuals()['fake']
        #print(result_image)
        result_image = util.tensor2im(result_image)
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = cv2.resize(result_image, size)
        if save_only:
            out.write(result_image)
        else:
            imshow("cam-output", result_image)

        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break
    webcam.release()
    if save_only:
        out.release()
    destroyAllWindows()
