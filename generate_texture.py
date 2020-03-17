# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import cv2
import argparse
import numpy as np
import os


class Generate_texture:
    def __init__(self, model_path):
        print(model_path)

        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.model.cuda()

    def generate_texture(self, img_path):
        img = cv2.imread(img_path)


        img = cv2.resize(img, (64, 128))
        img = (img / 225. - 0.5) * 2.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda()

        out = self.model(img)

        out = out.cpu().detach().numpy()[0]
        out = out.transpose((1, 2, 0))
        out = (out / 2.0 + 0.5) * 255.
        out = out.astype(np.uint8)
        out = cv2.resize(out, dsize=(64, 64))

        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show generated image')
    parser.add_argument('--gpu', '-g')
    parser.add_argument('--img', '-i', default='example_texture_generation/input')
    parser.add_argument('--model', '-m', default='texture_generation_weights/texture_generation_weights.pkl')
    parser.add_argument('--out', '-o', default='example_texture_generation/texture')

    args = parser.parse_args()
    img_path = args.img
    out_path = args.out
    model_path = args.model

    torch.nn.Module.dump_patches = True

    generator = Generate_texture(model_path)

    print(img_path)
    if os.path.isdir(img_path):
        for root, dir, names in os.walk(img_path):
            for name in names:
                full_path = os.path.join(img_path, name)
                print('executing: ', full_path)
                out = generator.generate_texture(img_path=full_path)

                print('finish: ', os.path.join(out_path, name))
                cv2.imwrite(os.path.join(out_path, name), out)
    else:
        out = generator.generate_texture(img_path=img_path)

        cv2.imshow('out', out)
        cv2.waitKey(0)
