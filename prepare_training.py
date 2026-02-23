from dataloader.dataset_loader import trainset_loader
from tqdm import tqdm
from multiprocessing import Pool


def process_index(index):
    trainset_loader.preprocess_to_bev(index)

    # single_rgb_bev, single_label_bev, rgb_bev, label_bev, full_rgb_bev, full_label_bev = trainset_loader.preprocess_from_bev(index)
    # fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(single_rgb_bev)
    # ax[1].imshow(single_label_bev)
    # ax[2].imshow(rgb_bev)
    # ax[3].imshow(label_bev)
    # ax[4].imshow(full_rgb_bev)
    # ax[5].imshow(full_label_bev)
    # plt.show()


if __name__ == "__main__":
    total = list(range(0, len(trainset_loader)))

    n_procces = 1
    with Pool(n_procces) as p:
        ret_list = list(tqdm(p.imap_unordered(process_index, total), total=len(total)))