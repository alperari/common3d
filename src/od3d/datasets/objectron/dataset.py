import logging


logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_SequenceDataset
from od3d.datasets.sequence_meta import OD3D_SequenceMetaCategoryMixin

from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaSizeMixin,
)
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame

# from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEOGORIES
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import shutil
from od3d.datasets.objectron.frame import Objectron_FrameMeta, Objectron_Frame
from od3d.datasets.objectron.sequence import Objectron_Sequence
from od3d.datasets.objectron.enum import (
    OBJECTRON_CATEGORIES,
    MAP_CATEGORIES_OBJECTRON_TO_OD3D,
)
import torch
from od3d.datasets.object import (
    OD3D_CAM_TFORM_OBJ_TYPES,
    OD3D_FRAME_MASK_TYPES,
    OD3D_MESH_TYPES,
    OD3D_MESH_FEATS_TYPES,
    OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
    OD3D_TFROM_OBJ_TYPES,
    OD3D_PCL_TYPES,
    OD3D_SEQUENCE_SFM_TYPES,
)
from od3d.datasets.frame import OD3D_FRAME_KPTS2D_ANNOT_TYPES, OD3D_FRAME_MODALITIES


class Objectron(OD3D_SequenceDataset):
    all_categories = list(OBJECTRON_CATEGORIES)
    sequence_type = (
        Objectron_Sequence  # od3d.datasets.monolmb.sequence.MonoLMB_Sequence
    )
    frame_type = Objectron_Frame  # od3d.datasets.monolmb.frame.MonoLMB_Frame

    def path_frames_rgb(self):
        return Objectron.get_path_frames_rgb(path_raw=self.path_raw)

    @staticmethod
    def get_path_frames_rgb(path_raw: Path):
        return path_raw.joinpath("frames")

    def get_frame_by_name_unique(self, name_unique):
        return self.frame_type(
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            name_unique=name_unique,
            all_categories=self.categories,
            mask_type=OD3D_FRAME_MASK_TYPES.SAM_BBOX,
            cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.META,
            mesh_type=OD3D_MESH_TYPES.CUBOID500,
            mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
            mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
            pcl_type=OD3D_PCL_TYPES.KEYPOINTS,
            sfm_type=OD3D_SEQUENCE_SFM_TYPES.DROID,
            modalities=self.modalities,
            kpts2d_annot_type=OD3D_FRAME_KPTS2D_ANNOT_TYPES.META,
            tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
        )

    def get_sequence_by_name_unique(self, name_unique):
        return self.sequence_type(
            name_unique=name_unique,
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            all_categories=self.categories,
            mask_type=OD3D_FRAME_MASK_TYPES.SAM_BBOX,
            cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.META,
            mesh_type=OD3D_MESH_TYPES.CUBOID500,
            mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
            mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
            pcl_type=OD3D_PCL_TYPES.KEYPOINTS,
            sfm_type=OD3D_SEQUENCE_SFM_TYPES.DROID,
            modalities=self.modalities,
            tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            kpts2d_annot_type=OD3D_FRAME_KPTS2D_ANNOT_TYPES.META,
        )

    @staticmethod
    def extract_meta(config: DictConfig):
        path_meta = Objectron.get_path_meta(config=config)
        path_raw = Path(config.path_raw)

        if path_meta.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous Objectron")
            shutil.rmtree(path_meta)

        path_meta.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def setup(config: DictConfig):
        import requests
        from od3d.datasets.objectron.enum import OBJECTRON_CATEGORIES

        from pathlib import Path

        path_objectron_raw = Path(config.path_raw)

        if path_objectron_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous Objectron")
            shutil.rmtree(path_objectron_raw)

        if path_objectron_raw.exists():
            logger.info(f"Found Objectron dataset at {path_objectron_raw}")
        else:
            logger.info(f"Downloading Objectron dataset at {path_objectron_raw}")

            logger.info("categories")
            for category in list(OBJECTRON_CATEGORIES):
                category_str = f"{category}"
                logger.info(category_str)

            for category in list(OBJECTRON_CATEGORIES):
                category_str = f"{category}"
                logger.info(category_str)
                public_url = "https://storage.googleapis.com/objectron"
                blob_path = public_url + f"/v1/index/{category_str}_annotations"
                video_ids = requests.get(blob_path).text
                video_ids = video_ids.split("\n")

                from tqdm import tqdm

                # Download the first ten videos in cup test dataset
                for i in tqdm(range(len(video_ids))):
                    video_id = video_ids[
                        i
                    ]  # "bike/batch-4/38" "bike/batch-4/43" # video_ids[i]
                    if len(video_id) == 0:
                        continue
                    video_filename = public_url + f"/videos/" + video_id + "/video.MOV"
                    metadata_filename = (
                        public_url + "/videos/" + video_id + "/geometry.pbdata"
                    )
                    annotation_filename = (
                        public_url + "/annotations/" + video_id + ".pbdata"
                    )

                    seq_name = "_".join(video_id.split("/")[1:]).replace("-", "_")

                    annotation = requests.get(annotation_filename)
                    path_objectron_raw.joinpath(category_str, seq_name).mkdir(
                        exist_ok=True, parents=True
                    )
                    annotation_fpath = path_objectron_raw.joinpath(
                        category_str, seq_name, "annotation.pbdata"
                    )
                    file = open(annotation_fpath, "wb")
                    file.write(annotation.content)
                    file.close()

                    metadata = requests.get(metadata_filename)
                    metadata_fpath = path_objectron_raw.joinpath(
                        category_str, seq_name, "meta.pbdata"
                    )
                    file = open(metadata_fpath, "wb")
                    file.write(metadata.content)
                    file.close()

                    # video.content contains the video file.
                    video = requests.get(video_filename)
                    video_fpath = path_objectron_raw.joinpath(
                        category_str, seq_name, "video.MOV"
                    )
                    file = open(video_fpath, "wb")
                    file.write(video.content)
                    file.close()

    @staticmethod
    def extract_meta(config: DictConfig):
        path_meta = Objectron.get_path_meta(config=config)
        path_raw = Path(config.path_raw)

        config.setup.remove_previous = True
        if path_meta.exists() and config.get("extract_meta", False).get(
            "remove_previous", False
        ):
            logger.info(f"Removing previous Objectron meta")
            shutil.rmtree(path_meta)

        path_meta.mkdir(parents=True, exist_ok=True)

        dict_nested_frames = config.get("dict_nested_frames", None)
        dict_nested_frames_banned = config.get("dict_nested_frames_ban", None)
        preprocess_meta_override = config.get("extract_meta", False).get(
            "override",
            False,
        )

        categories = (
            list(dict_nested_frames.keys())
            if dict_nested_frames is not None
            else OBJECTRON_CATEGORIES.list()
        )

        # path_sequences = Objectron.get_path_frames_rgb(path_raw=path_raw)
        # sequences_count_max_per_category = config.get(
        #    "sequences_count_max_per_category",
        #    None,
        # )

        for category in categories:
            logger.info(f"preprocess meta for class {category}")

            sequences_names = (
                list(dict_nested_frames[category].keys())
                if dict_nested_frames is not None
                and category in dict_nested_frames.keys()
                and dict_nested_frames[category] is not None
                else None
            )
            if sequences_names is None and (
                dict_nested_frames is None
                or (
                    category in dict_nested_frames.keys()
                    and dict_nested_frames[category] is None
                )
            ):
                if not path_raw.joinpath(category).exists():
                    continue

                sequences_names = [
                    fpath.stem for fpath in path_raw.joinpath(category).iterdir()
                ]
            if (
                dict_nested_frames_banned is not None
                and category in dict_nested_frames_banned.keys()
                and dict_nested_frames_banned[category] is not None
            ):
                sequences_names = list(
                    filter(
                        lambda seq: seq
                        not in dict_nested_frames_banned[category].keys(),
                        sequences_names,
                    ),
                )

            from tqdm import tqdm

            for sequence_name in tqdm(sequences_names):
                fpath_sequence_meta = OD3D_SequenceMetaCategoryMixin.get_fpath_sequence_meta_with_category_and_name(
                    path_meta=path_meta,
                    category=category,
                    name=sequence_name,
                )

                if fpath_sequence_meta.exists() and not preprocess_meta_override:
                    continue

                sequence_meta = OD3D_SequenceMetaCategoryMixin.load_from_raw(
                    category=category,
                    name=sequence_name,
                )

                if (
                    len(
                        list(
                            path_raw.joinpath(
                                sequence_meta.name_unique,
                            ).iterdir(),
                        ),
                    )
                    > 0
                ):
                    # filtering sequences with no rgb images
                    sequence_meta.save(path_meta=path_meta)

                path_sequence = path_raw.joinpath(
                    sequence_meta.name_unique,
                )
                frames_path = path_raw.joinpath("frames", category, sequence_name)
                fpath_video = path_sequence.joinpath("video.MOV")
                fpath_annotation = path_sequence.joinpath("annotation.pbdata")
                # fpath_meta_objectron = path_sequence.joinpath('meta.pbdata')

                from od3d.datasets.objectron.schema import (
                    annotation_data_pb2 as annotation_protocol,
                )

                with open(fpath_annotation, "rb") as pb:
                    sequence_annotation = annotation_protocol.Sequence()
                    sequence_annotation.ParseFromString(pb.read())

                import cv2

                # Create a VideoCapture object
                cap = cv2.VideoCapture(fpath_video)

                # Check if the video was opened successfully
                if not cap.isOpened():
                    print("Error: Could not open video.")
                    exit()

                frame_id = 0
                # Loop through the video frames
                while True:
                    # Read a frame
                    ret, frame = cap.read()

                    # If we got a valid frame, process it
                    if ret:
                        fpath_frame = frames_path.joinpath(f"{frame_id}.jpg")
                        fpath_frame.parent.mkdir(exist_ok=True, parents=True)
                        # Display the frame (optional)
                        # cv2.imshow('Frame', frame)
                        cv2.imwrite(filename=fpath_frame, img=frame)

                        frame_id += 1
                        # Wait for 25ms and check if the user pressed the 'q' key to exit
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break

                        l_size = torch.LongTensor(
                            [frame.shape[0], frame.shape[1]]
                        ).tolist()
                        frame_meta = Objectron_FrameMeta.load_from_raw(
                            name=fpath_frame.stem,
                            category=category,
                            sequence_name=sequence_name,
                            rfpath_rgb=Path(fpath_frame.relative_to(path_raw)),
                            l_size=l_size,
                            annotation=sequence_annotation,
                        )
                        if frame_meta is not None:
                            # could be none if there is no object annotation
                            frame_meta.save(path_meta=path_meta)

                    else:
                        # If no more frames, break the loop
                        break

                # Release the VideoCapture object and close any open windows
                cap.release()
                cv2.destroyAllWindows()
