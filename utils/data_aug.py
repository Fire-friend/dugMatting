import random
import kornia


def random_crop(*imgs):
    H_src, W_src = imgs[0].shape[2:]
    W_tgt = random.choice(range(640, 960)) // 32 * 32
    # H_tgt = random.choice(range(400, 600)) // 32 * 32
    H_tgt = W_tgt
    scale = max(W_tgt / W_src, H_tgt / H_src)
    results = []
    for img in imgs:

        img = kornia.resize(img, (int(H_src * scale), int(W_src * scale)), interpolation='nearest')
        img = kornia.center_crop(img, (H_tgt, W_tgt), mode='nearest')
        img = kornia.resize(img, (H_src, W_src), interpolation='nearest')

        # if torch.sum(torch.abs(img) - 1) == 0:
        #     img = kornia.resize(img, (int(H_src * scale), int(W_src * scale)), interpolation='nearest')
        #     img = kornia.center_crop(img, (H_tgt, W_tgt), mode='nearest')
        #     img = kornia.resize(img, (H_src, W_src), interpolation='nearest')
        # else:
        #     img = kornia.resize(img, (int(H_src * scale), int(W_src * scale)))
        #     img = kornia.center_crop(img, (H_tgt, W_tgt))
        #     img = kornia.resize(img, (H_src, W_src))
        results.append(img)
    return results