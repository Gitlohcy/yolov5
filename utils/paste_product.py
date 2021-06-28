from utils.gen_image import *
from utils.yolo_utils.file_util import read_color_imgs
from typing import List
import yaml
from pathlib import Path
from utils.yolo_utils.imgaug_util import stochastic_params, show_dist
from imgaug import parameters as iap


class PasteProduct:

    img_file_type = [".jpg", ".png", ".jpeg"]

    def __init__(
        self,
        hyp_dict,
        coco_path,
        back_img_path=None,
        front_img_path=None,
        cls_names: List[str] = None,
    ):
        self.front_imgs = {}

        if back_img_path:
            self.set_back_fnames(back_img_path)
        if front_img_path:
            self.set_front_fnames(front_img_path)

        self.set_coco(coco_path)
        self.set_coco_df()
        self.set_mask(cls_names)
        self.set_hyp_dict(hyp_dict)
        self.reset_augs()
        self.reset_data_dist()

    def set_back_fnames(self, back_img_path):
        self.back_img_path = Path(back_img_path)
        print("reading back_img paths...")
        self.back_fnames = pd.Series(
            fu.img_list_from(self.back_img_path, self.img_file_type)
        )

    def set_front_fnames(self, front_img_path):
        self.front_img_path = Path(front_img_path)
        print("reading front_img paths...")
        self.front_fnames = [
            f"{p.parent.name}/{p.name}"
            for p in fu.img_list_from(self.front_img_path, self.img_file_type)
        ]

    def set_mask(self, cls_names: List[str] = None):

        catIds = (
            self.coco.getCatIds(catNms=cls_names)
            if cls_names
            else self.coco.getCatIds()
        )
        annIds = self.coco.getAnnIds(catIds=catIds, iscrowd=None)
        self.anns = self.coco.loadAnns(annIds)
        self.anns_df = pd.DataFrame(self.anns)

        print("Converting polygon points to mask...")
        self.anns_df["mask"] = [self.coco.annToMask(ann) for ann in tqdm(self.anns)]

        if not self.anns_df["image_id"].is_unique:
            raise ValueError("All image_id must be unique")
        is_same_id = (self.anns_df.index.values == self.anns_df["image_id"]).all()
        self.get_mask = (
            self.get_mask_by_idx if is_same_id else self.get_mask_by_image_id
        )

    def set_coco(self, coco_path):
        self.coco_path = Path(coco_path)
        assert (
            self.coco_path.is_file() and self.coco_path.suffix == ".json"
        ), f"`{coco_path}` is not a valid json file"

        self.coco = COCO(coco_path)

    def set_coco_df(self):
        coco_json = boxx.loadjson(self.coco_path)
        boxx.tree(coco_json, deep=1)
        self.coco_df = {}
        self.coco_df["images"] = pd.DataFrame(coco_json["images"])
        self.coco_df["annotations"] = pd.DataFrame(coco_json["annotations"])
        self.coco_df["categories"] = pd.DataFrame(coco_json["categories"])

    def set_hyp_dict(self, hyp_dict):
        ## assign tuple for range between (min, max) in imgaug library
        keys_with_range = ["rotate", "batch_resize"]

        if "batch_resize_distribution" in hyp_dict and hyp_dict[
            "batch_resize_distribution"
        ] in ["normal", "uniform"]:

            hyp_dict["batch_resize"] = stochastic_params(
                *hyp_dict["batch_resize"], dist=hyp_dict["batch_resize_distribution"]
            )
            print(
                f"""resize in {hyp_dict["batch_resize_distribution"]} distribution,
                    range({hyp_dict['batch_resize']}) will be used as distribution's min max"""
            )
            print("plot resize distribution:")
            show_dist([hyp_dict["batch_resize"]])

            hyp_dict["rotate"] = tuple(hyp_dict["rotate"])
        else:
            for k in keys_with_range:
                hyp_dict[k] = tuple(hyp_dict[k])

        for c in "hsv":
            hyp_dict["hsv_" + c][1] = tuple(hyp_dict["hsv_" + c][1])

        self.hyp_dict = hyp_dict
        self.hyp_dict["m_blur_chance"] = hyp_dict["motion_blur"][0]
        self.hyp_dict["m_blur_kernel"] = tuple(hyp_dict["motion_blur"][1])
        self.hyp_dict["m_blur_angle"] = tuple(hyp_dict["motion_blur"][2])

    def reset_augs(self):
        self.adjust_hsv = iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="RGB",
            children=iaa.Sequential(
                [
                    iaa.Sometimes(
                        self.hyp_dict["hsv_h"][0],  # chance
                        color_chnl_adjust(0, self.hyp_dict["hsv_h"][1]),  # (min, max)
                    ),
                    iaa.Sometimes(
                        self.hyp_dict["hsv_s"][0],
                        color_chnl_adjust(1, self.hyp_dict["hsv_s"][1]),
                    ),
                    iaa.Sometimes(
                        self.hyp_dict["hsv_v"][0],
                        color_chnl_adjust(2, self.hyp_dict["hsv_v"][1]),
                    ),
                ]
            ),
        )

        self.aug_funcs_dict = {
            0: None,
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None,
            10: None,
            11: None,
            12: None,
            13: None,
            14: None,
        }

        self.motion_blur = iaa.MotionBlur(
            k=self.hyp_dict["m_blur_kernel"], angle=self.hyp_dict["m_blur_angle"]
        )

    def reset_data_dist(self):
        """Set the distribution for number of front img to paste in single back_img"""

        assert (
            len(self.hyp_dict["paste_dist"]) == self.hyp_dict["max_paste_n"]
        ), "length of paste_dist nid to be same with maximum paste num"

        self.n_products_dist = np.array(
            random.choices(
                range(1, self.hyp_dict["max_paste_n"] + 1),
                self.hyp_dict["paste_dist"],
                k=10000,
            )
        )  # ensure samples is big enough

    def get_mask_by_idx(self, idx):
        return self.anns_df.iloc[idx]["mask"]

    def get_mask_by_image_id(self, img_id):
        is_id = self.anns_df["image_id"] == img_id
        return self.anns_df[is_id]["mask"].values[0]

    def get_img_fname(self, img_id):
        is_id = self.coco_df["images"]["id"] == img_id

        if is_id.any():
            return self.coco_df["images"][is_id]["file_name"].values[0]
        else:
            return None

    def get_catId_by_image_id(self, img_id):
        is_id = self.anns_df["image_id"] == img_id
        return self.anns_df[is_id]["category_id"].values[0]

    def get_batch_img_ids(self, n, by="ann"):
        options = ["ann", "image"]

        if by == "ann":
            return self.anns_df["image_id"].sample(n)
        elif by == "image":
            return self.coco_df["images"]["id"].sample(n)
        else:
            raise ValueError(f"only {options} allowed")

    def next_batch(self, n):
        def check_len(list_, len_ids):
            if not len(list_) == len_ids:
                raise ValueError("Length of Value does not match length of img_ids")

        cls_ids, imgs, fnames, masks = [], [], [], []
        img_ids = self.get_batch_img_ids(n, by="image")
        for id in img_ids:
            fname = self.get_img_fname(id)
            if fname is None:
                print(f"file <{fname}> not found skip requested image")
                continue
            fnames.append(fname)
            masks.append(self.get_mask_by_image_id(id))
            cls_ids.append(self.get_catId_by_image_id(id))

            img = self.front_imgs.get(id, None)  # is cached
            imgs.append(
                img if img is not None else imageio.imread(self.front_img_path / fname)
            )

        [check_len(l, len(img_ids)) for l in [fnames, imgs, masks, cls_ids]]

        return fnames, imgs, masks, cls_ids

    def show_next_batch(self, n, show_mask=True):
        def draw_mask(img, mask):
            return iu.draw_mask(img, iu.get_segmap(img, mask))

        fnames, imgs, masks, cls_ids = self.next_batch(6)
        imgs_to_show = (
            [draw_mask(img, mask) for img, mask, in zip(imgs, masks)]
            if show_mask
            else imgs
        )

        pu.show_batch(imgs_to_show, titles=fnames)

    def filter_front_img(self, corrupted_imgs):
        """filter img on coco_df
        Args:
        ----
        corrupted_imgs: List[str], optional
            filtered the corrupted img that match the `file_name` field in coco_df
        Eg:
        [ 'h45__wangwang_coco_orange/0000028.jpg',
        'h45__wangwang_coco_orange/0000029.jpg']
        """

        if len(corrupted_imgs):
            is_corrupted = self.coco_df["images"]["file_name"].isin(corrupted_imgs)
            self.coco_df["images"] = self.coco_df["images"][~is_corrupted]

        fname_is_exist = self.coco_df["images"]["file_name"].isin(self.front_fnames)
        id_is_match = self.coco_df["images"]["id"].isin(self.anns_df["image_id"])
        img_is_chosen = fname_is_exist & id_is_match
        self.coco_df["images"] = self.coco_df["images"][img_is_chosen]

    def cache_imgs(self, which):
        if which == "back":
            self.back_imgs = {p.name: imageio.imread(p) for p in tqdm(self.back_fnames)}

        elif which == "front":
            # coco
            # self.back_imgs = {p.name: imageio.imread(p) for p in tqdm(self.back_fnames)}
            tqdm.pandas()
            fname_is_exist = self.coco_df["images"]["file_name"].isin(self.front_fnames)
            img_is_chosen = self.coco_df["images"][fname_is_exist]["id"].isin(
                self.anns_df["image_id"]
            )

            img_df = self.coco_df["images"]
            self.front_imgs = pd.Series(
                img_df["file_name"][img_is_chosen].progress_apply(
                    lambda fname: imageio.imread(self.front_img_path / fname)
                ),
                index=img_df["id"][img_is_chosen],
            ).to_dict()

        else:
            raise ValueError("only `back` or `front` is allowed")

    def paste_front_imgs(self, back_img, num2gen=1):
        pasted_back_img_list = []
        clean_bbs_list = []

        ## start generate
        n_products_per_img = random.choices(self.n_products_dist, k=num2gen)[0]

        fnames, imgs, masks, cls_ids = self.next_batch(n_products_per_img)
        img_segmap_cls = list(zip(*[imgs, masks, cls_ids]))

        # augmentation on small_imgs
        resize_seqs = [
            [
                batch_resize(self.hyp_dict["batch_resize"]),
                depth_resize,
            ]
            for depth_resize in depth_level_resize(
                len(img_segmap_cls), min_size=self.hyp_dict["depth_min_size"]
            )
        ]

        augmented_small_imgs = []
        for resize_seq, (img, segmap, cls_id) in zip(resize_seqs, img_segmap_cls):
            aug_seqs = [
                *resize_seq,
                iaa.Rotate(self.hyp_dict["rotate"]),
                self.adjust_hsv,
            ]
            aug_seqs = append_selective_aug(
                aug_seqs, cls_id, self.aug_funcs_dict, self.hyp_dict["class_aug"]
            )
            aug_seqs = iaa.Sequential(aug_seqs)
            augmented_img, augmented_segmap = aug_front_img(img, segmap, aug_seqs)
            augmented_small_imgs.append([augmented_img, augmented_segmap, cls_id])

        ## back_img padding
        # back_img = read_background_img(back_img_list)
        ws, hs = imgs_wh([img for img, _, _ in augmented_small_imgs])
        padded_back_img = pad_back_img_to_fixedSize(back_img, (ws.max(), hs.max()))

        ## generate coords
        rand_coords = get_rand_cell_coords(back_img, 10)
        ph, pw, _ = np.array(padded_back_img.shape) - np.array(back_img.shape)
        padded_rand_coords = rand_coords.copy()
        padded_rand_coords += [pw / 2, ph / 2]  # size of padding for each side

        ## pasting
        bboxes = []
        for i, (img, segmap, cls_id) in enumerate(augmented_small_imgs):
            bool_mask = segmap.get_arr() > 0
            padded_back_img, bbox = paste_into_img2(
                img, padded_back_img, padded_rand_coords[i], bool_mask
            )
            aug_during_paste(
                padded_back_img,
                bbox,
                bool_mask,
                self.motion_blur,
                active_chance=self.hyp_dict["m_blur_chance"],
            )
            bboxes.append([*bbox, cls_id])

        ## crop
        bbs = iu.get_bbs(padded_back_img, bboxes)
        bih, biw, _ = back_img.shape
        pasted_back_img, pasted_bbs = iaa.CenterCropToFixedSize(height=bih, width=biw)(
            image=padded_back_img, bounding_boxes=bbs
        )
        pasted_bbs = pasted_bbs.clip_out_of_image()

        ## remove overlap bbox
        bbox_to_remove = remove_overlaped_bbox(
            self.hyp_dict["overlap_thereshold"],
            augmented_small_imgs,
            padded_rand_coords,
            padded_back_img,
        )
        test_pasted_bbs = np.array(pasted_bbs.bounding_boxes)
        clean_bbs = BoundingBoxesOnImage(
            test_pasted_bbs[~bbox_to_remove], pasted_back_img
        )

        bboxes = bbs_to_yoloList(clean_bbs)

        return pasted_back_img, bboxes

    def gen_back_img(self, n):
        back_fnames = pd.Series(self.back_fnames)
        return (
            self.paste_front_imgs(back_img)
            for back_img in tqdm(read_color_imgs(back_fnames, n), total=n)
        )

    def gen_then_save(self, n, img_dest: Path, lbl_dest: Path):
        mkdir_if_notExist(Path(img_dest))
        mkdir_if_notExist(Path(lbl_dest))

        for img, bbox in self.gen_back_img(n):
            fu.write_img_and_bboxes(img, bbox, img_dest, lbl_dest)


def create_paste_instance(paste_data_yaml, paste_hyp_yaml, cache="front"):

    with open(str(paste_data_yaml)) as f:
        paste_data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    with open(str(paste_hyp_yaml)) as f:
        paste_hyp_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # input
    back_img_path = (
        Path(paste_data_dict["back_img"]) if "back_img" in paste_data_dict else None
    )

    front_img_path = Path(paste_data_dict["front_img"])
    coco_path = Path(paste_data_dict["coco_path"])

    if not (front_img_path.is_file() or front_img_path.is_dir()):
        raise ValueError("front_img_path is no a valid file or directory path")

    if not coco_path.is_file() or coco_path.suffix != ".json":
        raise ValueError("coco_path is no a valid json file")

    paste_p = PasteProduct(
        paste_hyp_dict,
        coco_path,
        front_img_path=front_img_path,
        back_img_path=back_img_path,
    )
    if cache in ["front", "back"]:
        paste_p.cache_imgs(cache)

    return paste_p